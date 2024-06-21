##########################################################################
# This document combines each of the individual components in the SBF 
# pipeline, in order to calculate the SBF amplitude.
# Also, the file structure is defined in this document.
##########################################################################

import sys
# path where the individual components are stored.
sys.path.append("C:/Users/Lei/Documents/Courses/MSc Astronomy/Thesis/MAIN/pipeline/functions")

import os # to handle file structures
from astropy.io import fits # opening and savind fits files
import numpy as np

from inspect import getsource # in order to be able to print the source code.

# Own functions
from extractdata import extractData 
from backgroundmodel import backgroundLevelAnalysis
from ellipsemodels import fitInitialEllipseModel, fitFinalEllipseModel
from sourcemasking import findInitialSourceMask, findFinalSourceMask
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas

# -----------------------------------------------------------------------------------------
# Additional functions required to obtain the output variables in one function.

def createRequiredVariables(data, model_final, source_mask_final, total_background):
    """
    From the data and the model, the nri, model mask, and total mask is returned.
    """
    mask_model = model_final <= 1.5 * total_background
    mask_combined = np.array(~(mask_model | source_mask_final), dtype=int)
    
    nri = (data - model_final)/np.sqrt(model_final)
    nri[np.isinf(nri)] = 0
    
    mask_combined = ~np.isnan(nri)&mask_combined
    
    nri[np.isnan(nri)] = 0

    nri *= mask_combined
    return mask_model, mask_combined, nri
       


# -----------------------------------------------------------------------------------------
# Functions that allow saving the file structure.
# Main function here is openCreateFileStructure

def openCreateFileStructure(folder_name,
                            file_names,
                            data_type,
                            file_path, bool_files,
                            function, *args, **kwargs):
    """
    Function that creates a file structure for the given files and folders
    
    folder_name: Must be a string, indicating the name of the folder (within
                 the file_path) in which the files must be stored.
    file_names:  A list with strings, indicating the names of the files to be 
                 created.
    data_type:   A list with the type of object each variable to be saved. Can 
                 either be: "array", "fits", "value", "mask", or "3d array"
    file_path:   The path in which the folder must be created.
    bool_files:  A list of three bools, for which each idx indicates:
                 idx 0: Must it be attempted to extract the file first.
                 idx 1: If it cannot be extracted, do we want to compute the 
                        variables via the given function.
                 idx 2: If the values are computed, must they be stored.
    function:    The function to be called when the variables still must be 
                 computed
    *args:       Input parameters for the function
    **kwargs:    Input parameters for the function
    """
    folder_path = file_path + "/" + folder_name
    extract_files, compute_files, save_files = bool_files
    
    if extract_files:
        # Check if the folder exists:
        if os.path.isdir(folder_path):
            variables = loadFiles(folder_path, file_names, data_type)
            return variables
    
    if compute_files:
        variables = function(*args, **kwargs)

        if save_files:
            createDirectory(folder_path, print_information=True)
            saveFiles(variables, folder_path, file_names, data_type)

        return variables

    return np.zeros(len(file_names))
    
    
def loadFiles(path, file_names, data_type):
    """
    Load the files in the given folder, following the given substructure
    and data types
    """
    variables = []
    
    for idx in range(len(file_names)):
        if data_type[idx] == "array":
            value = np.loadtxt(path + "/" + file_names[idx])
        elif data_type[idx] == "fits":
            value = openFits(path + "/" + file_names[idx])
        elif data_type[idx] == "value":
            value = np.loadtxt(path + "/" + file_names[idx], ndmin=1)[0]
        elif data_type[idx] == "mask":
            value = openMask(path + "/" + file_names[idx])
        elif data_type[idx] == "3d array":
            value = load3dArray(path + "/" + file_names[idx])
        else:
            print("Incorrect data type supplied")
            break
        variables.append(value)
        
    return variables


def saveFiles(variables, path, file_names, data_type):
    """
    Save the files in the given folder, following the given substructures 
    and data types
    """
    if len(file_names) == 1:
        variables = [variables]
    
    for idx in range(len(file_names)):
        if data_type[idx] == "array":
            np.savetxt(path + "/" + file_names[idx], variables[idx])
        elif data_type[idx] == "fits":
            saveFits(path + "/" + file_names[idx], variables[idx])
        elif data_type[idx] == "value":
            np.savetxt(path + "/" + file_names[idx], [variables[idx]])
        elif data_type[idx] == "mask":
            saveMask(path + "/" + file_names[idx], variables[idx])
        elif data_type[idx] == "3d array":
            save3dArray(path + "/" + file_names[idx], variables[idx])
            
        else:
            print("Incorrect data type supplied")
            break  
            
    return

def save3dArray(path, array):
    """
    Function that stores a 3d array as a folder, in which 
    each of the 2d arrays are saved as individual files.
    """
    createDirectory(path, print_information=False)
    for idx in range(len(array)):
        np.savetxt(path + "/file{}".format(idx+1), array[idx])
    return

def load3dArray(path):
    """
    Function that load a 3d array from a folder, in which 
    each of the 2d arrays are saved as individual files.
    """
    array_3d = []
    files = os.listdir(path)
    for idx in range(len(files)):
        array_2d = np.loadtxt(path + "/file{}".format(idx+1))
        array_3d.append(array_2d)
    return np.array(array_3d)
    
def saveFits(path, data):
    """
    Function checks whether a fits file exists.
    If it does, that file is removed.
    A new fits file is created with the data.
    """
    if os.path.exists(path):
        os.remove(path)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(path)
    return

def openFits(path):
    """
    Opens the first data content in a fits files
    """
    # memmap ensures the file is closed after being opened.
    with fits.open(path, memmap=False) as hdu:
        data = hdu[0].data
    hdu.close
    return data

def saveMask(path, mask):
    """
    Function that saves a mask as a fits file
    """
    mask = mask.astype(int)
    saveFits(path, mask)
    return

def openMask(path):
    """
    Function that opens a mask from a fits file
    """
    mask_fits = openFits(path)
    mask = mask_fits.astype(bool)
    return mask

def createDirectory(path, print_information=True):
    """
    Checks whether the specified path exists, if it doesn't then
    the directory is created.
    """
    if os.path.isdir(path) != True:
        os.mkdir(path)
        if print_information:
            print("Folder '"+ path + "' created in directory:")
            print(os.getcwd())
    return


# -----------------------------------------------------------------------------------------
# Redefining all subset functions in order to allow saving the file structure

"""
In each of the functions below, a folder is defined, as well as the names of the 
individual files that are to be stored in these folders.

The data_type corresponds to the different type of data that each file must be 
saved as. This can be either "array", "fits", "value", "mask", or "3d array"
"""


def extractData_(file_path, bool_files, *args, **kwargs):
    folder_name = "1.extract_data"
    file_names = ["data_combined.fits", "mask_cr.fits", 
                  "exptime.txt", "initial_backround.txt",
                  "cutout_mask.fits"]
    data_type = ["fits", "mask", "value", "value", "mask"]
    function = extractData
    print("\n1. Extracting the data ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


def backgroundLevelAnalysis_(file_path, bool_files, *args, **kwargs):
    folder_name = "2.iterate_background_noise"
    file_names = ["iterated_background_noise.txt", "data.fits", 
                  "total_background.txt", "std_background.txt"]
    data_type = ["array", "fits", "value", "value"]
    function = backgroundLevelAnalysis
    print("\n2. Estimating the background level ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


def fitInitialEllipseModel_(file_path, bool_files, *args, **kwargs):
    folder_name = "3.initial_ellipse_model"
    file_names = ["residual_basic.fits", "model_basic.fits"]
    data_type = ["fits", "fits"]
    function = fitInitialEllipseModel
    print("\n3. Fitting the initial ellipse model ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


def findInitialSourceMask_(file_path, bool_files, *args, **kwargs):
    folder_name = "4.initial_source_mask"
    file_names = ["source_mask.fits", "center_sources.fits"]
    data_type = ["mask", "mask"]
    function = findInitialSourceMask
    print("\n4. Find the initial source mask ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


def fitFinalEllipseModel_(file_path, bool_files, *args, **kwargs):
    folder_name = "5.final_ellipse_model"
    file_names = ["residual_final.fits", "model_final.fits"]
    data_type = ["fits", "fits"]
    function = fitFinalEllipseModel
    print("\n5. Fitting the final ellipse model ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


def findFinalSourceMask_(file_path, bool_files, *args, **kwargs):
    folder_name = "6.final_source_mask"
    file_names = ["source_mask.fits", "residual_power.txt", "sigma_residual_power.txt"]
    data_type = ["mask", "value", "value"]
    function = findFinalSourceMask
    print("\n6. Find the final source mask ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 

def createRequiredVariables_(file_path, bool_files, *args, **kwargs):
    folder_name = "7.main_variables"
    file_names = ["mask_model.fits", "mask_combined.fits", "nri.fits"]
    data_type = ["mask", "mask", "fits"]
    function = createRequiredVariables
    print("\n7. Compute the required variables ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 

def extractPsfSources_(file_path, bool_files, *args, **kwargs):
    folder_name = "8.psf_objects"
    file_names = ["psf_frames", "log_power_spectra.txt"]
    data_type = ["3d array", "array"]
    function = extractPsfSources
    print("\n8. Extract PSF objects and model PSF ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 

def calculateSBF_(file_path, bool_files, *args, **kwargs):
    folder_name = "9.sbf_components"
    file_names = ["image_ps.txt", "expected_ps.txt", "sbf.txt", "noise.txt"]
    data_type = ["array", "array", "value", "value"]
    function= calculateSBF
    print("\n9. Fit the SBF components")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 

def calculateLibrarySBF_(file_path, bool_files, *args, **kwargs):
    folder_name = "10.library_psf_analysis"
    file_names = ["library_psf", "expected_ps.txt", "library_sbf.txt", "library_noise.txt"]
    data_type = ["3d array", "array", "value", "value"]
    function = calculateLibrarySBF
    print("\n10. Calculating SBF signal with library psf ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 

def sbfMagnitudeAnnuliSigmas_(file_path, bool_files, *args, **kwargs):
    folder_name = "11.uncertainties"
    file_names = ["uncertainties.txt"]
    data_type = ["array"]
    function = sbfMagnitudeAnnuliSigmas
    print("\n11. Calculating the uncertainties ...")
    
    variables = openCreateFileStructure(folder_name,
                                        file_names,
                                        data_type,
                                        file_path, bool_files,
                                        function, *args, **kwargs)
    return variables 


# -----------------------------------------------------------------------------------------
# Defining the main pipeline

def sbfPipeline(data_path, file_path, bool_files, 
                file_type="flt", obs_filter="F160W",
                print_information=True, make_plots=True, return_variables=False, 
                lib_psf_peak_flux=0.2, save_images=True, image_path=None, 
                estimate_initial_background=True, gal_bckgr_frac=0.6,
                central_mask_radius=50):
    """
    Perform each individual step of the SBF pipeline. Store the files in the 
    given file_path. 
    
    bool_files corresponds to whether the files must be saved or not. See function
    openCreateFileStructure for details.
    """
    createDirectory(file_path, print_information)
    
    if save_images==True:
        if image_path == None:
            image_path=file_path + "/0.figures"
        createDirectory(image_path, print_information)

    data_combined, mask_cr, exptime, initial_bckgr, cutout_mask = extractData_(file_path, 
                                                       bool_files, data_path, file_type, 
                                                       estimate_initial_background)

    iterated_bckgr_level, data, total_bckgr, sigma_bckgr = backgroundLevelAnalysis_(
                                                       file_path, 
                                                       bool_files, data_combined, 
                                                       initial_bckgr, file_type=file_type, 
                                                       plot=make_plots, 
                                                       image_path=image_path)

    residual_basic, model_basic    = fitInitialEllipseModel_(file_path, bool_files, 
                                                       data, mask_cr=~mask_cr)

    source_mask, center_sources    = findInitialSourceMask_(file_path, bool_files, 
                                                       residual_basic, model_basic,
                                                       obs_filter, file_type, ~mask_cr,
                                                       plot=make_plots,
                                                       image_path=image_path,
                                                       image_title="4.1_initial_nri.png")

    residual_final, model_final    = fitFinalEllipseModel_(file_path, bool_files, 
                                                       data, source_mask, center_sources,
                                                       mask_cr=~mask_cr)

    source_mask_final, residual_power, sig_res_power = findFinalSourceMask_(file_path, 
                                                       bool_files,  residual_final, 
                                                       model_final,  obs_filter, 
                                                       file_type, ~mask_cr, total_bckgr,
                                                       gal_bckgr_fraction=gal_bckgr_frac,
                                                       central_mask_radius=central_mask_radius,
                                                       plot=make_plots,
                                                       image_path=image_path)

    mask_model, mask_combined, nri = createRequiredVariables_(file_path, 
                                                       bool_files, data, model_final, 
                                                       source_mask_final, total_bckgr)

    psf_frames, log_power_spectra  = extractPsfSources_(file_path, bool_files, 
                                                       obs_filter, data, model_final, 
                                                       data_path, cutout_mask, total_bckgr,
                                                       plot=make_plots, 
                                                       image_path=image_path)

    image_ps, expected_ps, sbf, noise = calculateSBF_(file_path, bool_files,
                                                       nri, mask_combined, psf_frames,
                                                       norm_type = "MaskedPixels",
                                                       fit_range_i=0.2, fit_range_f=0.6,  
                                                       plot=make_plots,
                                                       image_path=image_path)
    
    lib_psf, lib_psf_exp_ps, lib_psf_sbf, lib_psf_noise = calculateLibrarySBF_(file_path, 
                                                       bool_files, data_path, 
                                                       lib_psf_peak_flux, nri, 
                                                       mask_combined, obs_filter,
                                                       make_plots=make_plots,
                                                       image_path=image_path)
    
    uncertainties = sbfMagnitudeAnnuliSigmas_(file_path, bool_files, obs_filter, nri, 
                                                       mask_combined, psf_frames, lib_psf, 
                                                       sigma_bckgr, data, model_final, 
                                                       residual_power, kfit_i=0.2, kfit_f=0.7, 
                                                       plot=True, image_path=image_path)
    
    if return_variables == True:
        return [data_combined, mask_cr, exptime, initial_bckgr, iterated_bckgr_level, data, 
                total_bckgr, sigma_bckgr, residual_basic, model_basic, source_mask, 
                center_sources, residual_final, model_final, source_mask_final, 
                residual_power, sig_res_power, mask_model, mask_combined, nri, 
                psf_frames, log_power_spectra, image_ps, expected_ps, 
                sbf, noise, lib_psf, lib_psf_exp_ps, lib_psf_sbf, lib_psf_noise,
                uncertainties]
    else:
        return


# -----------------------------------------------------------------------------------------
# For convenience: print the functions within the SBF pipeline or alternatively, 
# print each of the returned variables

def printSbfVariables():
    print("""data_combined, mask_cr, exptime, initial_bckgr, iterated_bckgr_level, data, 
                total_bckgr, sigma_bckgr, residual_basic, model_basic, source_mask, 
                center_sources, residual_final, model_final, source_mask_final, 
                residual_power, sig_res_power, mask_model, mask_combined, nri, 
                psf_frames, log_power_spectra, image_ps, expected_ps, 
                sbf, noise, lib_psf, lib_psf_exp_ps, lib_psf_sbf, lib_psf_noise,
                uncertainties""")
    return

def printSbfPipeline():
    print(getsource(sbfPipeline))
    return
