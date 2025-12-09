##########################################################################
# Pipeline for SBF amplitude calculation, specifically for Euclid data, 
# based on the code of Lei Titulaer

# Version 2. Implementing unity in geometry
##########################################################################

version = "2.1"

# Imports 

import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from astropy.io import fits
from astropy.wcs import WCS 
from astroscrappy import detect_cosmics

from mgefit.find_galaxy import find_galaxy


import sys
sys.path.append("./functions")

# Own function imports 
from plotting import imdisplay
from extractdata import extractData
from backgroundmodel import mainBackgroundEstimation
from ellipsemodels import MainFitEllipseModel, fitApertureModel
from sourcemasking import maskBackgroundSources, maskCircle, createRequiredVariables
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF, appSBFmagnitude
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas


# ---------------------------------


###############
# Main SBF calculation pipeline function
###############


def MainPipeline(data_path, file_path=None, image_path=None, 
                 make_plots=True, plot_plots=True, 
                 psf=None, geometry=None, maxarea_sourcemask=None, filter="VIS", 
                 background_estimation=False, background=None, ellipsefitter='v1', cosmic_ray_method='astroscrappy'):
    """
    Combining the original with the new functions
    """
    print(f"version {version}")
    print(f'computing galaxy from {data_path}')
    print("\n1. Extracting the data ...")

    data, mask_cr, wcs, mzp = extractData(data_path, file_path=file_path, image_path=image_path, make_plots=make_plots, plot_plots=plot_plots, filter=filter, cosmic_ray_method=cosmic_ray_method)


    if background_estimation:
        print("\n2. Performing background estimation ...")
        data, total_bckgr, sex_bckgr = mainBackgroundEstimation(data, mask_cr, image_path=image_path, make_plots=make_plots, plot_plots=plot_plots)

    else:
        print("\n2. No background estimation performed ...")
        sex_bckgr = 0
        if background != None:
            total_bckgr = background
            data -= total_bckgr
        else:
            total_bckgr = 0

    if file_path != None:
        np.savetxt(file_path + "/background", [total_bckgr, sex_bckgr])


    if ellipsefitter == 'a1':
        print("\n5. Fitting aperture ellipse model ...")
        residual_final, model_final, geometry = fitApertureModel(data, mask_cr=mask_cr, geometry=None, make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path)

    else:
        print("\n3. Fitting initial ellipse model ...")
        residual_basic, model_basic, geometry = MainFitEllipseModel(data, mask_cr=mask_cr, make_plots=make_plots, plot_plots=plot_plots, final=False, method=ellipsefitter)

        print("\n4. Finding initial source mask ...")
        source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, make_plots=make_plots, plot_plots=plot_plots, maxarea=maxarea_sourcemask, r=3)

        print("\n5. Fitting final ellipse model ...")
        residual_final, model_final, geometry = MainFitEllipseModel(data, mask_cr=source_mask, geometry=geometry, make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path,method=ellipsefitter)

    print("\n6. Finding final source mask ...")
    source_mask_final = maskBackgroundSources(residual_final, mask_cr=mask_cr, make_plots=make_plots, plot_plots=plot_plots, maxarea=maxarea_sourcemask, image_path=image_path, final=True, original_image=data, r=3)
 
    print("\n7. Creating required variables ...")
    mask_model, mask_combined, nri = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, geometry, make_plots=make_plots, plot_plots=plot_plots, image_path=image_path)
    
    print("\n8 Calculate power spectra  ...")
    image_ps, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, psf,
                                                       norm_type = "MaskedPixels",
                                                       fit_range_i=0.1, fit_range_f=0.6,  
                                                       make_plots=make_plots,plot_plots=plot_plots,
                                                       image_path=image_path)
    print("\n9. Calculate sbf magnitude")
    sbfmag = appSBFmagnitude(sbf, mzp)

    if file_path != None:
        np.savetxt(file_path + "/data_background_subtracted", data)
        np.savetxt(file_path + "/combined_final_mask", mask_combined)
        np.savetxt(file_path + "/final_model", model_final)
        geometry_params = np.array((geometry.x0, geometry.y0, geometry.sma, geometry.eps, geometry.pa))
        np.savetxt(file_path + "/geometry", geometry_params)

    return sbf, sbfmag, total_bckgr, geometry.sma, mzp




