import numpy as np
from astropy.io import fits

from os import listdir

from fourierfunctions import imageToPowerSpectrum, sizePsf
from empiricalpsf import plotPsfPowerSpectra, getMedianBackground
from fourierfunctions import calculateSBF

def maskCentralSquare(image, npix):
    """
    Function returns a mask of size image, with the border of width npix
    equal to 1 and the remaining center 0
    """
    mask = np.ones(image.shape, dtype=bool)
    mask[npix:-npix, npix:-npix] = 0
    return mask


def openFitsPsf(filename):
    """
    Open a fits file using astropy.fits
    """
    with fits.open(filename) as hdu:
        data = hdu[0].data
    
    return data


def processLibraryPsf(filename, npix_background=18, framesize=51):
    """
    Given a PSF file from the library, process it by
    1. Open the file 
    2. Estimate background 
    3. Assign mean background to nan values
    4. subtract background
    5. Normalise psf
    """
    psf_raw = openFitsPsf(filename)                                # 1.
    mean_background = getMedianBackground(psf_raw, npix_background)# 2.
    psf_raw[np.isnan(psf_raw)] = mean_background                   # 3.
    psf_raw -= mean_background                                     # 4.
    psf_clean = psf_raw / np.sum(psf_raw)                          # 5.
    
    return psf_clean


def findOutlierIdx(psf, threshold, mask):
    """
    Find the indexes of the outliers in a psf image.
    """
    psf_temp = np.copy(psf)
    psf_temp[mask] = np.nan
    
    mean = np.nanmean(psf_temp)
    std = np.nanstd(psf_temp)
    
    idx_outlier_pos = psf_temp > mean + threshold * std 
    idx_outlier_neg = psf_temp < mean - threshold * std
    
    return idx_outlier_pos | idx_outlier_neg 


def selectPsf(psf, peak_flux=0.2, npix_std=18, std_max=0.002, 
              std_threshold=10, n_outlier=0):
    """
    Function that selects the psf based on a number of criteria:
    1. The peak pixel value must be above a fixed value peak_flux
    2. The standard deviation must be below a fixed value std_max
    3. There must be no outliers further than 10std in the background
    """
    # create the mask where the central square is equal to 0
    center_mask = maskCentralSquare(psf, npix_std)
    
    flag_1 = np.max(psf) > peak_flux
    flag_2 = np.std(psf[center_mask]) < std_max
    flag_3 = len(psf[findOutlierIdx(psf, std_threshold, ~center_mask)]) <= n_outlier
    
    return flag_1 & flag_2 & flag_3


def reshapePsfCutouts(psf_list):
    """
    Reshape each of the psf's in a cutout to the smallest size in the list
    Necessary because some cutouts from the HST archive are not equally 
    shaped to 51 pix size.
    
    Assumes square, centered, unevenly sized PSF cutouts.
    """
    psf_list_new = []
    smallest_shape = min([psf.shape[0] for psf in psf_list])
    
    for idx in range(len(psf_list)):
        if psf_list[idx].shape[0] == smallest_shape:
            psf_list_new.append(psf_list[idx])
        elif psf_list[idx].shape[0] != smallest_shape:
            shape_idx = psf_list[idx].shape[0]
            delta_shape = shape_idx - smallest_shape
            idx_cut = int(delta_shape/2)
            
            psf_cutout = psf_list[idx][idx_cut:-idx_cut, idx_cut:-idx_cut]
            psf_list_new.append(psf_cutout/np.sum(psf_cutout))
    
    return psf_list_new


def extractLibraryPsf(folder, npix_background=18, peak_flux=0.2, 
                             npix_std=18, std_max=0.002, std_threshold=10,
                             n_outlier=0, plot=False, image_path=None):
    """
    Open all the fits files in a folder, compute the psf,
    select based on criteria and return the array of psf frames. 
    """
    psf_selected = []
    
    for file in listdir(folder):
        psf = processLibraryPsf(folder + "/" + file, npix_background)
        if selectPsf(psf, peak_flux, npix_std, std_max, std_threshold):
            psf_selected.append(psf)
            
    psf_selected = reshapePsfCutouts(psf_selected)
    
    if plot==True:
        image_title = None
        if image_path != None:
            image_title = image_path+"/10.1_library_psf_sources.png"
        plotPsfPowerSpectra(psf_selected, image_path=image_title)
    
    return np.array(psf_selected)


#########################################################################################
# Main function IR
#########################################################################################

def calculateLibrarySbfIR(data_path, peak_flux,
                       nri, mask_combined, make_plots=True,
                       image_path=None):
    """
    First, a check is made whether the folder with library PSFs exists, if it does
    then the sbf amplitude given the library psf is estimated.
    """
    if ("psf_files" in listdir(data_path)):
        library_psf = extractLibraryPsf(data_path + "/psf_files", peak_flux=peak_flux,
                                       plot=make_plots, image_path=image_path)
        _, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, library_psf,
                                                norm_type = "MaskedPixels",
                                                fit_range_i=0.15, fit_range_f=0.7,
                                                plot=make_plots, image_path=image_path, 
                                                image_title="10.2_sbf_fit_library_psf.png")
                                                  #kfit_i=50, kfit_f=-1, plot=make_plots)
        return library_psf, expected_ps, sbf, noise
    elif ("psf_files" in listdir(data_path + "\\..")):  #!this way of creating a path does now work
        library_psf = extractLibraryPsf(data_path + "\\.." + "/psf_files",
                                        peak_flux=peak_flux,
                                        plot=make_plots, image_path=image_path)
        _, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, library_psf,
                                                norm_type = "MaskedPixels",
                                                fit_range_i=0.15, fit_range_f=0.7,
                                                plot=make_plots, image_path=image_path, 
                                                image_title="10.2_sbf_fit_library_psf.png")
        return library_psf, expected_ps, sbf, noise
    else:
        print("No library psf object folder available in folder: \n"+ data_path)
        return [[0]], [[0]], 0, 0
    
    
#########################################################################################
# Main functions UVIS
#########################################################################################

# Same as in sbf_pipeline
def load3dArray(path):
    """
    Function that load a 3d array from a folder, in which 
    each of the 2d arrays are saved as individual files.
    """
    array_3d = []
    files = listdir(path)
    for idx in range(len(files)):
        array_2d = np.loadtxt(path + "/file{}".format(idx+1))
        array_3d.append(array_2d)
    return np.array(array_3d)


def calculateLibrarySbfUVIS(data_path, peak_flux,
                       nri, mask_combined, obs_filter, make_plots=True,
                       image_path=None):
    """
    Extract the psf objects from a given folder.
    """
    if obs_filter == "F475W":
        psf_path = data_path + "/../../../PSF475W/psf_files_475"
    elif obs_filter == "F850LP":
        psf_path = data_path + "/../../../PSF850LP/psf_files_850"
    else:
        print("No library psf sources available.")
        return [[0]], [[0]], 0, 0
    
    psf_files = load3dArray(psf_path) 
    
    _, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, psf_files,
                                              norm_type = "MaskedPixels",
                                              fit_range_i=0.2, fit_range_f=0.6,
                                              plot=make_plots, image_path=image_path,
                                              image_title="10.2_sbf_fit_library_psf.png")
    return psf_files, expected_ps, sbf, noise
    

#########################################################################################
# Main Function
#########################################################################################

def calculateLibrarySBF(data_path, peak_flux,
                       nri, mask_combined, obs_filter, make_plots=True,
                       image_path=None):
    """
    Function that calculates the SBF amplitude by the library PSF
    """
    if (obs_filter == "F160W") or (obs_filter == "F110W"):
        return calculateLibrarySbfIR(data_path, peak_flux,
                       nri, mask_combined, make_plots, image_path)
    else:
        return calculateLibrarySbfUVIS(data_path, peak_flux,
                       nri, mask_combined, obs_filter, make_plots, image_path)
    
    
    
    


            