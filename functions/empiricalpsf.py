import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

import sep
import os

from tqdm.notebook import tqdm

import sys
# sys.path.append("C:/Users/Lei/Documents/Courses/MSc Astronomy/Thesis/MAIN/pipeline")
# sys.path.append("C:/Users/Lei/Documents/Courses/MSc Astronomy/Thesis/MAIN/pipeline/functions")


from fourierfunctions import imageToPowerSpectrum, sizePsf
from sourcemasking import obtainSextractorSources
from extractdata import extractFileList


def maskCircle(data, x0, y0, rout, rin=0):
    """
    Mask a circular annulus with given center, inner and outer radius. 
    
    The area between the inner and outer radius is returned as True values.
    """
    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    
    shape = data.shape
    
    xrange = np.arange(max(0,x0-rout),min(shape[1],x0+rout), dtype=int)
    yrange = np.arange(max(0,y0-rout),min(shape[0],y0+rout), dtype=int)

    mask = np.zeros(shape, dtype=bool)
    
    for x in xrange:
        for y in yrange:
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            if (r <= rout) & (r >= rin):
                mask[y,x] = True
                
    return mask

# ### Note: Same as in sbf_pipeline
# def createDirectory(path, print_information=True):
#     """
#     Checks whether the specified path exists, if it doesn't then
#     the directory is created.
#     """
#     if os.path.isdir(path) != True:
#         os.mkdir(path)
#         if print_information:
#             print("Folder '"+ path + "' created in directory:")
#             print(os.getcwd())
#     return


def getMedianBackground(image, npix):
    # ! make the same mistake of taking corners point without checking theres actual data there
    """
    estimate the mean value in an image by extracting boxes of width npix from
    the corners of the image, estimating the mean, and returning the median.
    """
    val_1 = np.nanmean(image[-npix:, -npix:])
    val_2 = np.nanmean(image[-npix:, :npix])
    val_3 = np.nanmean(image[:npix, -npix:])
    val_4 = np.nanmean(image[:npix, :npix])
    
    return np.median([val_1, val_2, val_3, val_4])


def frameCutout(image, x_center, y_center, boxwidth):
    """
    Returns a cutout from the original image, centered at the input coordinates
    and with given boxwidth. 
    
    If no full cutout could be made due to the center coordinates being too 
    close to the border of the original image, a box of zeroes is returned.
    
    Also flag is returned, indicating whether the cutout was successful.
    """
    
    img_xsize, img_ysize = image.shape
    x1 = int(round(x_center - boxwidth/2))
    x2 = int(round(x_center + boxwidth/2))
    
    y1 = int(round(y_center - boxwidth/2))
    y2 = int(round(y_center + boxwidth/2))
    
    if x1>0 and y1>0 and x2<img_xsize and y2<img_ysize:
        return np.copy(image[x1:x1+boxwidth, y1:y1+boxwidth]), 1
    else:
        return np.zeros([boxwidth, boxwidth]), 0
    
        
def processAndNormalisePsf(psf, mask=None, npix=18):
    """
    - Substitutes gaussian noise at the positions of the mask. 
    - Subtracts background level
    - Normalises total psf to one
    """
    if type(mask) != type(None):
        psf[mask] = np.nan
    
    psf[np.isnan(psf)] = generateMockData(psf, len(psf[np.isnan(psf)]), npix)
    
    psf -= getMedianBackground(psf, npix)

    psf /= np.sum(psf)
    return psf


def generateMockData(psf, len_data, npix=None):
    """
    Estimate the mean and standard deviation of the non-masked pixels in the 
    psf frame. Then generate a list of len_data random Gaussian generated 
    data points with estiamted mean and standard deviation.    
    """
    psf_temp = np.copy(psf)
    
    if npix != None:
        psf_temp[npix:-npix, npix:-npix] = np.nan
        
    mean = np.nanmean(psf_temp)
    std = np.nanstd(psf_temp)
    
    return np.random.normal(mean, std, len_data)


def maskOutliersPsf(psf, mask, thr=3):
    """
    Returns a mask of all sources extracted with SexTractor in the psf frame,
    taking into account the image mask. The inverse objects are also masked.
    """
    _, segmap = obtainSextractorSources(psf, mask, thr, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=1, max_area=None)
    
    _, segmap_inv = obtainSextractorSources(-psf, mask, thr, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=1, max_area=None)
    
    
    return (segmap!=0) | (segmap_inv!=0)


def maskSigmaOutliers(psf, circle_mask, threshold=3):
    """
    Identify all pixels that have value of threshold*sigma from the mean.
    Substitute Gaussian noise for these pixels.
    """
    psf_temp = np.copy(psf)
    psf_temp[circle_mask] = np.nan
    
    mean = np.nanmean(psf_temp)
    std = np.nanstd(psf_temp)    
    
    idx = (psf_temp > mean + threshold*std) | (psf_temp < mean - threshold*std)
    
    psf[idx] = generateMockData(psf_temp, len(psf[idx]))
    return psf


def obtainDaofindSources(image, mask, sigma=3.0, fwhm=3.0, thr=15):
    """
    Extract point sources using the photutils DaoStarfinder tool. The chosen
    default parameters are such that the tool optimally selects potential psf
    stars
    
    Returned object is a Daofind table.
    
    Conventions are taken from the DAOfind documentation:
    https://photutils.readthedocs.io/en/stable/detection.html
    """
    
    mean, median, std = sigma_clipped_stats(np.ma.masked_array(image, mask), 
                                            sigma=sigma)  

    daofind = DAOStarFinder(fwhm=fwhm, threshold=thr*std)  

    sources = daofind(image * ~mask) 
    return sources


def selectDaofindSources(mask, sources, cutout_size=51):
    """
    Return an array of x- and y coordinates of the objects that adhere to
    the conditions based on the distance to image border and number of masked
    pixels in the frame
    """
    good_sources = []
    for source in sources:
        x0, y0 = source["xcentroid"], source["ycentroid"]
        # Note, y0 and x0 are turned around for frameCutout
        mask_cutout, flag = frameCutout(mask, y0, x0, cutout_size)
        if (flag == 1) and (np.sum(mask_cutout) < 0.02*cutout_size**2):
            good_sources.append([x0, y0])
    return np.array(good_sources)

#################### Main Sources ########################

def identifyPsfSources(data, galaxy_model, cutout_size=51):
    """
    Identify all point sources in the original image frame and return
    the variables necessary for the further psf calculation.
    """
    residual_image = data-galaxy_model
    model_mask = galaxy_model == 0
    
    obj, segmap = obtainSextractorSources(residual_image, model_mask, 
                                          thr=3, min_area=150)
    
    obj_inv, segmap_inv = obtainSextractorSources(-residual_image, model_mask, 
                                          thr=3, min_area=150)
    
    mask_extended_sources = (segmap != 0) | (segmap_inv != 0)
    
    mask_combined = model_mask | mask_extended_sources
    
    all_sources = obtainDaofindSources(residual_image, mask_combined)
    sources = selectDaofindSources(mask_combined, all_sources, cutout_size=51)
    
    return sources, mask_combined, residual_image


def processPsf(image, mask, x0, y0, circle_mask=None, cutout_size=51, 
               n_pix_normalise=18, sigma_threshold=3):
    """
    Cutout a psf and process it such that the background does not anymore 
    contain background sources.
    """
    if type(circle_mask) == type(None):# if not specified, mask circle of 
                                       # radius 1/8 of boxsize
        circle_mask = maskCircle(np.zeros([cutout_size,cutout_size]), 
                                 int(cutout_size/2), int(cutout_size/2), 
                                 int(cutout_size/8)) 
    
    psf_raw, _ = frameCutout(image, y0, x0, cutout_size)
    mask0,   _ = frameCutout(mask, y0, x0, cutout_size)
    
    psf_scaled = processAndNormalisePsf(psf_raw, mask=mask0, npix=n_pix_normalise)
    
    psf = np.copy(psf_scaled) # Object which all alterations are being done to
    psf_masked = np.copy(psf) # Object being masked for background processing
    psf_masked[circle_mask] = np.nan
    
    
    mask_outliers = maskOutliersPsf(psf, circle_mask)
    psf_masked[mask_outliers] = np.nan

    psf[mask_outliers] = generateMockData(psf_masked, len(psf[mask_outliers]))
    
    psf = maskSigmaOutliers(psf, circle_mask, threshold=sigma_threshold)
    
    psf -= np.nanmean(psf[~circle_mask])     # last subtract of backgr.
    psf /= np.sum(psf)                       # and normalise again
    return psf


def selectPsf(psf, circle_mask, std_threshold=0.003, max_pixel_flux=0.2):
    """
    Apply selection criteria in order to determine whether the psf is of 
    sufficient quality to be included for final psf calculation
    """
    std = np.std(psf[~circle_mask])
    obj = obtainSextractorSources(psf, np.zeros(psf.shape), 3, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=2, max_area=None)[0]
    obj_inv = obtainSextractorSources(-psf, np.zeros(psf.shape), 3, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=1, max_area=None)[0]
    if len(obj) == 0:
        return False
    
    flag_1 = std < std_threshold
    flag_2 = len(obj) == 1
    flag_3 = obj["npix"][0] > 20
    flag_4 = obj["flux"][0] > 0.9 and obj["flux"][0] < 1.1
    flag_5 = len(obj_inv) == 0
    flag_6 = np.max(psf) > max_pixel_flux
    
    return flag_1 & flag_2 & flag_3 & flag_4 & flag_5 & flag_6


#########################################################################################
# Functions required for UVIS analysis
#########################################################################################



def extractPsfUVISOuter(data_path, cutout_mask, total_background=0,
                        max_pixel_flux=0.15, std_threshold_selection = 0.001,
                        plot=True, image_path=None):
    """
    Extract the psf sources from the outer part of a drizzled HST image. 
    """
    # Check for background level, if it is 0 give it a very low value
    if total_background == 0:
        total_background = 1e-7
        
    # extract full data frame from folder
    files = extractFileList(data_path)
    data_raw = files[0][0]
    
    # generate background model -> to be subtracted for raw data and format for psf extractor
    background_model = np.array(~cutout_mask, dtype=int) * total_background
    
    psf_frames, log_power_spectra = extractPsfObjects(data_raw, background_model, 
                                             max_pixel_flux=max_pixel_flux,
                                             std_threshold_selection = std_threshold_selection,
                                             plot_selected_sources=plot, image_path=image_path)
    return psf_frames, log_power_spectra


def extractPsfObjectsUVIS(data, galaxy_model, data_path, cutout_mask, total_background, 
                          max_pixel_flux, std_thresh, plot, image_path=None, 
                          norm_factor=1):
    """
    Function that extracts the psf objects for an ACS image in the UVIS.
    
    Both sources in the inner of the image frame (in which the galaxy was modelled), as 
    wel as in the outer field of view are considered.
    """
    if image_path != None:
        image_path_inner = image_path + "/8.1_inner_psf"
        image_path_outer = image_path + "/8.1_outer_psf"
        createDirectory(image_path_inner)
        createDirectory(image_path_outer)
    else:
        image_path_inner, image_path_outer = None, None

    print("Extracting the inner PSF sources...")
    psf_frames_inner, log_power_spectra_inner = extractPsfObjects(data, galaxy_model, 
                                                     max_pixel_flux=max_pixel_flux,
                                                     std_threshold_selection=std_thresh,   
                                                     plot_selected_sources=plot,
                                                     image_path=image_path_inner)
    
    print("Extracting the outer PSF sources...")
    psf_frames_outer, log_power_spectra_outer = extractPsfUVISOuter(data_path, 
                                                      cutout_mask, total_background, 
                                                      max_pixel_flux, 
                                                      std_thresh, plot,
                                                      image_path=image_path_outer)
    
    if ((len(psf_frames_inner)==0) & (len(psf_frames_outer)==0))&(norm_factor>0.1):
        return extractPsfSources("F850LP", data, galaxy_model, data_path, 
                      cutout_mask, total_background, plot=plot,
                      image_path=image_path, norm_factor=norm_factor*0.5)
    elif len(psf_frames_inner)==0:
        psf_frames = psf_frames_outer
        log_power_spectra = log_power_spectra_outer
    elif len(psf_frames_outer)==0:
        psf_frames = psf_frames_inner
        log_power_spectra = log_power_spectra_inner
    else:
        psf_frames = np.vstack([psf_frames_inner, psf_frames_outer])
        log_power_spectra = np.vstack([log_power_spectra_inner, log_power_spectra_outer])
    
    if plot:
        plotPsfSample(psf_frames, print_parameters=True, print_selected=False,
                      image_path = image_path+"/8.2_zoom_psf_sources.png")
        plotPsfPowerSpectra(psf_frames, image_path = image_path+"/8.3_psf_power_spectra.png")
        
    return np.array(psf_frames), np.array(log_power_spectra)



############### semi-main function ##################

def extractPsfObjects(data, galaxy_model, cutout_size=51, 
                      n_pix_normalise=18, sigma_threshold_processing=3,
                      std_threshold_selection=0.003, max_pixel_flux=0.2,
                      plot_selected_sources=False,
                      plot_selected_PSFs=False,
                      plot_all_sources=False,
                      plot_psf_power_spectra=False,
                      image_path=None):
    """
    Function selects and extracts the suitable psf frames of a data frame 
    following specific extraction and selection steps.
    
    One array with the individual psf frames is returned and one with the 
    log power spectra of the individual PSF's.
    """
    sources, mask, res_image = identifyPsfSources(data, galaxy_model, 
                                                  cutout_size)
    psf_frames = []
    all_psf_frames = []
    
    log_psf_power_spectra = []
    
    # Gererate an annular mask for the psf processing
    circle_mask = maskCircle(np.zeros([cutout_size,cutout_size]), 
                                 int(cutout_size/2), int(cutout_size/2), 
                                 int(cutout_size/8)) 
    
    for x0, y0 in tqdm(sources):
        psf = processPsf(res_image, mask, x0, y0, circle_mask, cutout_size,
                         n_pix_normalise, sigma_threshold_processing)
        all_psf_frames.append(psf)

        if selectPsf(psf, circle_mask, std_threshold_selection, max_pixel_flux):
            psf_frames.append(psf)
            
            psf_large = sizePsf(psf, np.zeros([500,500]))
            power_spec = imageToPowerSpectrum(psf_large)
            log_psf_power_spectra.append(np.log10(power_spec))
            
    
#     if plot_all_sources==True:
#         plotPsfSample(all_psf_frames, print_parameters=True, circle_mask=circle_mask, 
#                       print_selected=True, 
#                       select_psf_params=[std_threshold_selection, max_pixel_flux])
        
#     if plot_selected_sources==True:
#         plotPsfSample(psf_frames, print_parameters=True, circle_mask=circle_mask,
#                       print_selected=False)
    
    if len(psf_frames) == 0:
        print("No suitable PSF sources found.")
        return [], []
    
    
    while len(np.array(psf_frames).shape) < 3:
        psf_frames = [psf_frames]

#     if plot_psf_power_spectra == True:
#         plotPsfPowerSpectra(psf_frames)
    makeRequiredPlots(plot_selected_sources, plot_selected_PSFs, plot_all_sources,
                      plot_psf_power_spectra, sources, mask, res_image, all_psf_frames, 
                      psf_frames, circle_mask, [std_threshold_selection, max_pixel_flux],
                      image_path=image_path)    
    
    return np.array(psf_frames), np.array(log_psf_power_spectra)

#########################################################################################
################################### Main functions ######################################
#########################################################################################


def returnPsfParams(obs_filter):
    """
    Function returns required parameters for psf modeling, given the used HST filter
    
    This function is important for calibration purposes.
    """
    if obs_filter == "F110W":
        max_pixel_flux, std_threshold_selection, filter_type = 0.175, 0.002, "IR"
    elif obs_filter == "F160W":
        max_pixel_flux, std_threshold_selection, filter_type = 0.1, 0.002, "IR"
    elif obs_filter == "F475W":
        max_pixel_flux, std_threshold_selection, filter_type = 0.12, 0.001, "UVIS"
    elif obs_filter == "F850LP":
        max_pixel_flux, std_threshold_selection, filter_type = 0.12, 0.001, "UVIS"
    else:
        print("No filter input, default parameters max_flux=0.1 and std=0.003 are used")
        max_pixel_flux, std_threshold_selection, filter_type = 0.1, 0.003, "IR"
    return max_pixel_flux, std_threshold_selection, filter_type
    

def extractPsfSources(obs_filter, data, galaxy_model, data_path="", 
                      cutout_mask=None, total_background=0, plot=True,
                      image_path=None, norm_factor=1):
    """
    Function that extracts PSF sources, given the observation type.
    
    If no sources are found, the function is re-run with a new norm factor.
    """
    max_pixel_flux, std_thresh, filter_type = returnPsfParams(obs_filter)
    std_thresh /= norm_factor
    max_pixel_flux *= norm_factor*2
    
    if filter_type == "IR":
        return extractPsfObjects(data, galaxy_model, max_pixel_flux=max_pixel_flux,
                                 std_threshold_selection=std_thresh,
                                 plot_selected_sources=plot, plot_selected_PSFs=plot,
                                 plot_all_sources=plot, plot_psf_power_spectra=plot,
                                 image_path=image_path)
    elif filter_type == "UVIS":
        return extractPsfObjectsUVIS(data, galaxy_model, data_path, cutout_mask, 
                                     total_background, max_pixel_flux, std_thresh, plot,
                                     image_path=image_path, norm_factor=norm_factor)
    else:
        return [[0]], [0]
    




#########################################################################################
# functions for plotting 
#########################################################################################

def plotSelectedSources(sources, mask, res_image, image_path=None):
    """
    Make a plot of the scaled residual image, while encircling the x and y 
    coordinates listed in the sources array.    
    """
    fig = plt.figure(figsize=[10,10])
    frame = fig.add_subplot(111)
    
    masked_image = np.ma.masked_array(res_image, mask)
    
    mean, median, std = sigma_clipped_stats(masked_image, sigma=3)  

    plt.imshow(masked_image, vmin=mean-10*std, vmax=mean+10*std)
    plt.scatter(sources[:,0], sources[:,1], s=80, facecolors='none', edgecolors='r')
    
    if image_path != None:
        plt.savefig(image_path)
    plt.show()
    return


def plotPsfSample(psf_list, print_parameters=True, circle_mask=None, print_selected=True, 
                  select_psf_params=[], image_path=None):
    """
    Make a plot of the psf frames in the list. The psf list must be an array of 
    individual psf cutouts
    
    The function plots a max of 64 frames in the list.
    """
    
    if type(circle_mask) == type(None):
            cutout_size = psf_list[0].shape[0]
            circle_mask = maskCircle(np.zeros([cutout_size,cutout_size]), 
                                 int(cutout_size/2), int(cutout_size/2), 
                                 int(cutout_size/8)) 
    
    n_array = min(8, int(np.sqrt(len(psf_list))+1))
    
    
    fig = plt.figure(figsize=np.array([20,20])*n_array/8)
    
    for idx in range(min(n_array**2, len(psf_list))):
        frame = fig.add_subplot(n_array, n_array,idx+1)

        psf0 = psf_list[idx]
        
        frame.imshow(psf0, vmin=-0.05, vmax=0.3, origin=1)

        psf_large = sizePsf(psf0, np.zeros([500,500]))
        ps_large = imageToPowerSpectrum(psf_large)
        frame.plot(np.linspace(0, 50, 250), 20*np.log10(ps_large)+30, c="yellow", alpha=0.5)

        if print_parameters==True:
            
            std = np.std(psf0[~circle_mask])
            obj = obtainSextractorSources(psf0, np.zeros(psf0.shape), 3, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=2, max_area=None)[0]
            obj_inv = obtainSextractorSources(-psf0, np.zeros(psf0.shape), 3, 
                                    box_width=64, subtract_sep_bckgr=True,
                                    min_area=1, max_area=None)[0]
            if len(obj) == 0:
                frame.text(3,9, "No Sextractor obj found", c="white")
            else:
                frame.text(3,9, "PSF flux = {:.2f}".format(obj["flux"][0]), c="white")
                frame.text(3,12, "N(objects) = {}".format(len(obj) + len(obj_inv)), c="white")
                frame.text(3,15, "PSF n_pix = {}".format(obj["npix"][0]), c="white")
            
            frame.text(3,3, "std = {:.5f}".format(std), c="white")
            frame.text(3,6, "peak flux = {:.3f}".format(np.max(psf0)), c="white")

        if (print_selected==True) and (selectPsf(psf0, circle_mask, *select_psf_params)):
            frame.text(3, 45,"Selected", c="red")
            
        frame.set_ylim(-0.5,50.5)
        frame.set_axis_off()
    plt.tight_layout()
    
    if image_path != None:
        plt.savefig(image_path)
    plt.show()
    return


def plotPsfPowerSpectra(psf_list, image_path=None):
    """
    Make a plot of the individual power spectra in a list of PSF's.
    
    Each individual power spectrum is plot, the power spec of the summed PSF,
    as well as the mean power spectrum of the individual PSF frames.
    """
    fig = plt.figure(figsize=[10,7])
    frame = fig.add_subplot(111)
    
    mean_psf = sum(psf_list)/np.sum(psf_list)
    
    log_ps = []

    for psf in psf_list:
        power_spec = imageToPowerSpectrum(sizePsf(psf, np.zeros([500,500])))
        log_ps.append(np.log10(power_spec))
        frame.plot(power_spec, c="0.5", alpha=0.5)
        
    frame.plot(10**np.mean(log_ps, axis=0), c="red", label="Average Power Spectrum")
    frame.plot(imageToPowerSpectrum(sizePsf(mean_psf, np.zeros([500,500]))),
               c="blue", label="Power Spectrum Average PSF")

    frame.grid()
    frame.legend(fontsize=14)
    frame.set_yscale("log")

    frame.set_xlim(0,250)

    frame.set_xlabel("Wave number (k)", fontsize=14)
    frame.set_ylabel("log(power)", fontsize=14)
    
    if image_path!= None:
        plt.savefig(image_path)
    plt.show()
    return


def makeRequiredPlots(plot_selected_sources,
                      plot_selected_PSFs,
                      plot_all_sources,
                      plot_psf_power_spectra,
                      sources, mask, res_image,   # For selected stars plot
                      all_psf_frames, psf_frames,  # For psf grid (whole or selected)
                      circle_mask, select_psf_params,
                      image_path=None
                     ):
    image_title = np.copy(image_path)
    if plot_selected_sources==True:
        if image_path != None:
            image_title = image_path + "/8.1_selected_psf_sources.png"
        plotSelectedSources(sources, mask, res_image, image_path=image_title)
    
    if plot_all_sources==True:
        if image_path != None:
            image_title = image_path + "/8.2_zoom_all_extracted_psf_sources.png"
        plotPsfSample(all_psf_frames, print_parameters=True, circle_mask=circle_mask, 
                      print_selected=True, 
                      select_psf_params=select_psf_params,
                      image_path=image_title)
        
    if plot_selected_PSFs==True:
        if image_path != None:
            image_title = image_path + "/8.3_zoom_selected_psf_sources.png"
        plotPsfSample(psf_frames, print_parameters=True, circle_mask=circle_mask,
                      print_selected=False, image_path=image_title)
    
    if (len(psf_frames) != 0) and (plot_psf_power_spectra == True):
        if image_path != None:
            image_title = image_path + "/8.4_psf_power_spectra.png"
        plotPsfPowerSpectra(psf_frames, image_path=image_title)

    return