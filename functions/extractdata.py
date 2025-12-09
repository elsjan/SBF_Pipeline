import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS   # In order to get the shift for overlaying frames

from astroscrappy import detect_cosmics
import lacosmic 

import os

from mgefit.find_galaxy import find_galaxy

from sourcemasking import obtainSextractorSources
from plotting import imdisplay


# -------------------------------------------------------------------------------

def openFits(file_path):
    """
    Open a SINGLE fits file, extract the data,
    magnitude_zero point, and wcs object (required for position information)
    Not always is the mzp in this keyword
    """
    for file in os.listdir(file_path):
        if ".fits" in file:
            with fits.open(file_path+ "/" + file) as hdu:
                data = hdu[0].data
                wcs = WCS(hdu[0].header)
                mzp = hdu[0].header['ZP_STACK']
                hdu.close()
    return data, wcs, mzp

def maskBrightCentralStars(frame, idx_border=25):
    """
    Identify bright stars in the frame, within a border of 25 pix
    from the outside of the frame
    """
    border_mask = np.zeros(frame.shape, dtype=bool)
    border_mask[idx_border:-idx_border,idx_border:-idx_border] = 1
    
    _, segmap = obtainSextractorSources(frame, border_mask, 
                                        thr=50, max_area=50)
    
    star_mask = segmap != 0
    
    return star_mask
    
    
def maskBadPixelsLacosmic(data_frame, effective_gain=2.3, readnoise=20):
    """
    Identify and mask the position of cosmic rays and bad pixels 
    (in an HST image) using te lacosmic procedure in combination with
    sextractor (sep). 
    1. First, we mask bright stars in the central area
    2. Then the negative outliers are identified.
    3. Cosmic rays are identified in the resulting image
    """
    data_frame = data_frame.byteswap().newbyteorder() # some necessary conversion
    
    star_mask = maskBrightCentralStars(data_frame)
                        
    # Parameters for lacosmic are typical for HST observations
    clean_frame, mask_neg = lacosmic(-data_frame, mask=star_mask,
                                      contrast=4, cr_threshold=4.5,
                                      neighbor_threshold=0.3,
                                      effective_gain=effective_gain,
                                      readnoise=readnoise)
    
    
    _, mask_pos = lacosmic(-clean_frame, mask=star_mask, 
                            contrast=4, cr_threshold=4.5, 
                            neighbor_threshold = 0.3, 
                            effective_gain=effective_gain, 
                            readnoise=readnoise)
    
    bad_pixel_mask = mask_neg | mask_pos
    
    return bad_pixel_mask


def maskBadPixelsAstroscrappy(data_frame, effective_gain=3.1, readnoise=4.5, filter="VIS", mask_filename=None):
    """
    Identify and mask remaining bad pixels in Euclid VIS ERO images.
    Euclid images are already flat-fielded and cosmic-ray cleaned,
    so thresholds are gentler than HST defaults.
    """
    
    # Optional: mask bright stars if relevant
    try:
        star_mask = maskBrightCentralStars(data_frame)
    except Exception:
        star_mask = np.zeros_like(data_frame, dtype=bool)
    
    # Gentle cosmic-ray detection
    if filter == "VIS":
        mask_cr, clean_frame = detect_cosmics(
            data_frame,
            gain=effective_gain,
            readnoise=readnoise,
            sigclip=6.0,
            sigfrac=0.5,
            objlim=5.0,
            satlevel=65535.0,  # Euclid VIS saturation (approx)
            verbose=False
        )
    elif filter == "H":
        mask_cr, clean_frame = detect_cosmics(
            data_frame,
            gain=effective_gain,
            readnoise=readnoise,
            sigclip=5.0,
            sigfrac=0.4,
            objlim=4.0,
            satlevel=1.118e5,  # Euclid NIR saturation (approx)
            verbose=False
        )
    else:
        print("Reminder to set the NIR filter parameters correctly!")
    bad_pixel_mask = mask_cr | star_mask

    # optional: mask known bad pixel regions from Euclid consortium
    # mask_euclid = fits.getdata(mask_filename).astype(bool)
    # bad_pixel_mask = bad_pixel_mask | mask_euclid
    
    return bad_pixel_mask





#############################################################################
# Main function
#############################################################################

    
def extractData(data_path, file_path=None, image_path=None, filter="VIS", cosmic_ray_method='astroscrappy',make_plots=True, plot_plots=True):
    """
    New version of extractData function
    """
        # make sure file and image path exist 
    if file_path != None:
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

    if image_path != None:
        try:
            os.makedirs(image_path)
        except FileExistsError:
            pass
    data, wcs, mzp = openFits(data_path)
    mask_nan = np.isfinite(data)
    if cosmic_ray_method == 'astroscrappy':
        mask_cr = maskBadPixelsAstroscrappy(data, filter=filter)
    elif cosmic_ray_method == 'lacosmic':
        mask_cr = maskBadPixelsLacosmic(data, filter=filter)
    mask_cr = mask_cr | ~mask_nan
    
    if make_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(data, ax, percentlow=1, percenthigh=99, scale='asinh')
        plt.title("Raw data")
        image_title = "1.1_raw_data.png"
        if image_path != None:
            plt.savefig(image_path + "/" + image_title)
        if plot_plots:
            plt.show()
        plt.close()

    return data, mask_cr, wcs, mzp
    


###############################################################################
# Old unused functions
###############################################################################

def meanIntensity(data, mask):
    """ 
    Function that calculates the mean intensity of an array. 
    
    Also a weight is returned, indicating the number of pixels that are not 
    masked.
    
    Mask is in the form that each redundant pixel is set to 0.
    
    Function is used for calculating the background level of a number of pixels 
    in the corner of a frame
    """ 
    print("small check", np.shape(data), np.shape(mask))
    print(data, mask)
    mean = np.median(data[~mask])
    weight = np.size(data[~mask])
    
    if np.isnan(mean): 
        mean = 0
        
    return mean, weight


def findBackground(data, mask, wSqr=5, corners=None):
    """
    Function that estimates the background level by taking four squares 
    in the corner of a frame, where each square has width wSqr, estimating 
    the average intensity in these corners, and subtracting that from the 
    original image.
    
    The masked pixels are not taken into account for the average intensity 
    calculation.
    """
    # for each of the four corners, a mean (mX) and a weight (wX) is calculated 
    # (depending on the number of non-masked pixels in that corner)
    # if corners == None:
    plt.figure()
    plt.imshow(data, norm='log')
    plt.show()
    m1, w1 = meanIntensity(data[:wSqr , :wSqr ], mask[:wSqr , :wSqr ])
    m2, w2 = meanIntensity(data[:wSqr , -wSqr:], mask[:wSqr , -wSqr:])
    m3, w3 = meanIntensity(data[-wSqr:, :wSqr ], mask[-wSqr:, :wSqr ])
    m4, w4 = meanIntensity(data[-wSqr:, -wSqr:], mask[-wSqr:, -wSqr:])

    # else:
    #     m1, w1 = meanIntensity(data[corners[0,0]:corners[0,0]+wSqr,corners[0,1]:corners[0,1]+wSqr]
    #                            , mask[corners[0,0]:corners[0,0]+wSqr, corners[0,1]:corners[0,1]+wSqr])
    #     m2, w2 = meanIntensity(data[corners[1,0]:corners[1,0]+wSqr,corners[1,1]:corners[1,1]+wSqr]
    #                            , mask[corners[1,0]:corners[1,0]+wSqr, corners[1,1]:corners[1,1]+wSqr])
    #     m3, w3 = meanIntensity(data[corners[2,0]:corners[2,0]+wSqr,corners[2,1]:corners[2,1]+wSqr]
    #                            , mask[corners[2,0]:corners[2,0]+wSqr, corners[2,1]:corners[2,1]+wSqr])
    #     m4, w4 = meanIntensity(data[corners[3,0]:corners[3,0]+wSqr,corners[3,1]:corners[3,1]+wSqr]
    #                            , mask[corners[3,0]:corners[3,0]+wSqr, corners[3,1]:corners[3,1]+wSqr])
    
    if w1 + w2 + w3 + w4 == 0:
        return 0 # no background level could accurately be found here.
    else:
        background_level = (w1*m1 + w2*m2 + w3*m3 + w4*m4)/(w1 + w2 + w3 + w4)
        
        return background_level


# Functions for opening .flt files

def extractFileList(folder):
    """
    Open files and extract information from all the fits files in a folder
    """
    data = []
    exptime = []
    wcs_obj = []

    for file in os.listdir(folder):
        if ".fits" in file:
            dat, exp, wcs = openFits(folder + "/" + file)
            data.append(dat)
            exptime.append(exp)
            wcs_obj.append(wcs)
        
    return data, exptime, wcs_obj

def overlapFrames(frame1, frame2, x0_1, y0_1, x0_2, y0_2):
    """
    Take frame1 and frame2, overlap the two such that
     - index x0_1 of frame1 overlaps with index x0_2 of frame 2
     - index y0_1 of frame1 overlaps with index y0_2 of frame 2
     - the maximum possible image in which each pixel contains 
       both frames is returned
     - Return the coordinates of the reference pixel in the new frame
    """
    size_x1, size_y1 = frame1.shape
    size_x2, size_y2 = frame2.shape
    
    lower_x = int(min(x0_1, x0_2))
    upper_x = int(min(size_x1 - x0_1, size_x2 - x0_2))
    
    lower_y = int(min(y0_1, y0_2))
    upper_y = int(min(size_y1 - y0_1, size_y2 - y0_2))

    cutout_1 = frame1[x0_1-lower_x:x0_1+upper_x, y0_1-lower_y:y0_1+upper_y]
    cutout_2 = frame2[x0_2-lower_x:x0_2+upper_x, y0_2-lower_y:y0_2+upper_y]    
    
    return cutout_1, cutout_2, lower_x, lower_y


def findAndApplyIntegerPixelShifts(data, mask, wcs, exptime, max_frame_size=1200):
    """
    Function iterates over the data and mask, and determines the pixel 
    shifts necessary for overlapping the frames. Both the mask as well 
    as the data are overlapped. data, mask, and wcs must be lists of the
    same length. 
    
    ref_xy is the index of the reference coords on the "old" frame
    new_xy is the index of the reference coords on the "new" frame
    
    The reference coords are the coordinates wo which each frame will be 
    centered before overlapping
    
    Each frame is weighed by the exposure time for that frame.
    """
    # Initiate the frame and mask
    combined_frame = np.zeros([max_frame_size, max_frame_size])
    combined_mask  = np.zeros([max_frame_size, max_frame_size], dtype=bool)
    
    for idx in range(len(data)):
        if idx==0: # initialising the coordinates in the center of the two frames
            ref_x, ref_y = int(max_frame_size/2), int(max_frame_size/2)
            new_x, new_y = int(len(data[idx][:,0])/2), int(len(data[idx][0,:])/2)
            reference_coords = wcs[idx].pixel_to_world(new_x, new_y) 
        
        new_y, new_x = [int(np.round(coord)) for coord in 
                        wcs[idx].world_to_pixel(reference_coords)]
    
        combined_mask, mask_idx, _, _ = overlapFrames(combined_mask, mask[idx], 
                                                      ref_x, ref_y, new_x, new_y)
        
        combined_frame, data_idx, ref_x, ref_y = overlapFrames(combined_frame, data[idx], 
                                                         ref_x, ref_y, new_x, new_y)
        
        combined_mask = combined_mask | mask_idx
        combined_frame = combined_frame + data_idx * exptime[idx]/np.sum(exptime)
    
    return combined_frame, combined_mask


def maximumSquareCutout(data):
    """
    Resize the data such that the maximum possible square cutout is returned,
    centered at the original frame.
    """
    xsize, ysize = data.shape
    size = min(data.shape)
    idx_x1 = int(xsize/2 - size/2)
    idx_x2 = int(xsize/2 + size/2)
    idx_y1 = int(ysize/2 - size/2)
    idx_y2 = int(ysize/2 + size/2)

    return data[idx_x1:idx_x2,idx_y1:idx_y2] 


def preProcessDataFlt(data, wcs, exptime, max_frame_size=1200):
    """
    Given the data, wcs object, and a max_frame size, pre process the
    data by 
     - Estimating the background in the corners of the image
     - finding and masking the bad pixels and cosmic rays in the frame
     - overlaying the frames following integer pixel shifts
    The combined frame, mask, and background level are returned. 
    """
    mask = []
    background = []
    for idx in range(len(data)):
        print("Identify bad pixels for {} / {} files ...".format(idx+1, len(data)))
        mask.append(maskBadPixels(data[idx]))
        background.append(findBackground(data[idx], 
                           mask[idx])*exptime[idx]/np.sum(exptime))
    
    data_combined, mask_combined = findAndApplyIntegerPixelShifts(data, 
                                 mask, wcs, exptime, max_frame_size=max_frame_size)
        
    data_square = maximumSquareCutout(data_combined)
    mask_square = maximumSquareCutout(mask_combined)
    
    return data_square, mask_square, background


def extractDataFlt(folder, estimate_background=False):
    """
    - Extract all the data from a folder. Folder can only contain .flt fits files
    - Process the data such that background is subtracted and bad sources masked
    - Return backgr. subtracted data, mask, total exp time, and backgr level
    """
    data, exptime, wcs = extractFileList(folder)
    
    data_combined, mask_combined, background = preProcessDataFlt(data, wcs, exptime)
    
    if estimate_background:
        data_subtracted = data_combined - np.sum(background)
    else: 
        data_subtracted = data_combined
        background = [0]
    return data_subtracted, ~mask_combined, np.sum(exptime), np.sum(background)


# Functions required for drizzled HST images.

def findCutoutCenter(data, gal_x, gal_y, cutout_size):
    """
    Given a data frame and gal center, return the center out of which a cutout
    must be made to create the maximally gal-centered cutout.
    """
    # find center pixel x and y
    center_x, center_y = int(data.shape[0]/2), int(data.shape[1]/2)
    
    delta_x = center_x - int(cutout_size/2)
    delta_y = center_y - int(cutout_size/2)
    
    if abs(gal_x-center_x)>delta_x:
        cutout_center_x = center_x - np.sign(gal_x-center_x) * delta_x
    else:
        cutout_center_x = gal_x
    
    if abs(gal_y-center_y)>delta_y:
        cutout_center_y = center_y - np.sign(gal_y-center_y) * delta_y
    else:
        cutout_center_y = gal_y
        
    return cutout_center_x, cutout_center_y

def maximumSquareCutoutGalcenter(data, mask, max_size=1500):
    """
    Make a square cutout of the data frame, given a not-square input. 
    
    The cutout frame is centered at the galaxy center, but an as large as
    possible square cutout will still be returned.
    """
    cutout_size = min(max_size, min(data.shape))
    dx, dy = int(cutout_size/2), int(cutout_size/2)
    
    f = find_galaxy(np.ma.masked_array(data, ~mask), plot=False, quiet=True)
    gal_x, gal_y = f.xpeak, f.ypeak
    
    cutout_center_x, cutout_center_y = findCutoutCenter(data, gal_x, gal_y, 
                                                        cutout_size)
        
    # Create a mask that indicates the part of the original file that was cut out
    mask_cutout = np.zeros(data.shape, dtype=bool)
    (mask_cutout[cutout_center_x-dx:cutout_center_x+dx,
                 cutout_center_y-dy:cutout_center_y+dy]) = 1
        
    return (data[cutout_center_x-dx:cutout_center_x+dx,
                 cutout_center_y-dy:cutout_center_y+dy], 
            mask[cutout_center_x-dx:cutout_center_x+dx,
                 cutout_center_y-dy:cutout_center_y+dy],
            mask_cutout)


def preProcessDataDrz(data_raw, estimate_background=False):
    """
    Pre-processes a raw fits file, in which bad pixels are identified as nan.
    
    The data and mask are cutout to a square image, which are returned together
    with an initial background estimate.
    """

    mask_raw = ~np.isnan(data_raw)
    
    data, mask, cutout_mask = maximumSquareCutoutGalcenter(data_raw, mask_raw)
    
#     if np.mean(data_raw.shape) < 1500: # for large files find_galaxy is inefficient
#         data, mask = maximumSquareCutoutGalcenter(data_raw, mask_raw)
#     else:
#         data, mask = maximumSquareCutoutGalcenter(data_raw, mask_raw, max_size)

    if estimate_background:
        background = findBackground(data, ~mask)
    else:
        background = 0
    
    # set nan values to 0
#     data[np.isnan(data)] = 0
    
    return data, mask, [background], cutout_mask


def extractDataDrz(folder, estimate_background=False):
    """
    - Extract a drizzeled data file from a folder
    - Process the data such thatthe image is square, and background is subtracted
    - Return backgr. subtracted data, mask, total exp time, and backgr level
    """
    files = extractFileList(folder)
    data_raw = files[0][0]
    exptime = files[1]
    
    data_combined, mask_combined, background, cutout_mask = preProcessDataDrz(data_raw,
                                                                    estimate_background)
    data_subtracted = data_combined - np.sum(background)
    return [data_subtracted, mask_combined, np.sum(exptime),
            np.sum(background), cutout_mask]




def OLDextractData(folder, file_type="flt", estimate_background=False):
    """
    - Extract all the data from a folder.
    - Process the data such that background is subtracted and bad sources masked
    - Return backgr. subtracted data, mask, total exp time, and backgr level
    
    filetype is either "flt" or "dr". 
    """
    if file_type == "flt":
        data_subtracted, mask_combined, exptime, background = extractDataFlt(folder,
                                                                estimate_background)
        cutout_mask = np.ones(data_subtracted.shape, dtype=bool)
        return data_subtracted, mask_combined, exptime, background, cutout_mask
    elif file_type == "dr":
        return extractDataDrz(folder, estimate_background)
    else:
        print("Error: filetype must be either 'flt' or 'dr'. Please  try again.")
        return [[0], [0], 0, 0, [0]]