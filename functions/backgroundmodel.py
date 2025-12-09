##########################################################################
# This document contains all the functions necessary to perform the 
# background estimation fitting a Sérsic profile to an ellipse fit
# The function iterateBackgroundNoise is the main function of the file.
##########################################################################

import numpy as np
from tqdm import tqdm  # changed to not-notebook version

import sep
# mgefit find_galaxy: To find the peak & orientation of a galaxy in an image
from mgefit.find_galaxy import find_galaxy

# In order to interpolate nan values
from pandas import DataFrame

# Functions required to fit the Sérsic model
from astropy.modeling.models import Sersic1D
from scipy.optimize import curve_fit

# Own functions from the ellipse class
from ellipsemodels import obtainEllipseInstance, obtainIsolistInstance

# In order to make plots
from matplotlib.pyplot import figure, show, savefig


# -----------------------------------------------------------------------------------------
# Functions required for modelling the background noise level

def sersicModelAstropy(r, n, I_e, r_e):
    """
    Function that returns the radial Sérsic intensity, using the Astropy Sersic1D function
    """
    func = Sersic1D(amplitude=I_e, r_eff=r_e, n=n)
    return func(r)


def sersicAndNoiseModel(r, noise, n, I_e, r_e):
    """
    1D Sérsic profile with an additional noise level. 
    """
    sersic_profile = sersicModelAstropy(r, n, I_e, r_e)
    return np.log10(sersic_profile + noise)


def findBackgroundNoise(data, fit_number, sclip=0, nclip=0, plot=False, 
                        frame=None, plot_number=None):
    """
    1. Function fits an ellipse isophote list to the data.
    2. A radial profile is fit.
    3. Background is determined by the radial profile
    
    fit_number corresponds to the number of outer isophotes used for the fit.
    """
    ellipse = obtainEllipseInstance(data)
    
    try: # Sometimes ellipse does not work directly, then sma_normfactor must be adjusted
        isolist = obtainIsolistInstance(ellipse, isolist_type="background")
    except:
        ellipse = obtainEllipseInstance(data, sma_normfactor=10)
        isolist = obtainIsolistInstance(ellipse, isolist_type="background")
    
    if len(isolist)==0:
        return 0
    
    # only consider the points where the intensity > 0
    idx_positive = isolist.intens > 0 
    
    fit_number = min(fit_number, len(isolist))
    
    f = find_galaxy(np.ma.masked_array(data, np.isnan(data)), plot=False, quiet=True)
    p0 = [0, 4, data[f.xpeak, f.ypeak]/1000, f.majoraxis] # initial parameter guess
    
    while fit_number > 0.5*len(isolist): # iterate until a fit can be made
        try:
            popt, pcov = curve_fit(sersicAndNoiseModel, 
                                   isolist.sma[idx_positive][-fit_number:], 
                                   np.log10(isolist.intens[idx_positive][-fit_number:]), 
                                   p0=p0)
            noise_level = popt[0]
            break
        except:
            fit_number -= 1
            popt = [0,0,0,0]
            noise_level = 0
    
    if plot==True:
        addFrameBackgroundModelling(frame, isolist, noise_level, popt, plot_number)
    
    return noise_level


def iterateBackgroundNoise(data, max_iter=10, fit_number=20, make_plots=False, plot_plots=True, 
                           initial_background=0, n_std=6, image_path=None):
    """
    Function iterates over the galaxy anumber of times, after which
    the background level gets subtracted after each iteration.
    
    Returned is the array of background levels found after each iteration. 
    If the updated background is 0 (no isolist could be fit), the algorithm stops.
    """

    bckgr_level = [0]
    if make_plots:
        n_array = np.ceil(np.sqrt(max_iter))
        fig = figure(figsize=[19,15])
        subplots = [fig.add_subplot(int(n_array), int(n_array), idx+1) for idx in range(max_iter)]
    else:
        subplots = np.zeros(max_iter) # need for a placeholder
        
    for idx in tqdm(range(max_iter)):
        bckgr_level_idx = findBackgroundNoise(data-np.sum(bckgr_level), 
                                              fit_number, plot=make_plots, 
                                              frame=subplots[idx], plot_number=idx+1)
        if bckgr_level_idx==0:
            break
        bckgr_level.append(bckgr_level_idx)
    
    if make_plots:
        if image_path != None:
            savefig(image_path + "/2.1_iterated_sersic_fits.png")
        if plot_plots:
            show()
        
    return bckgr_level


def interpolateNanValues(data_frame):
    """
    Linearly interpolate nan values using a pandas method. 
    """
    df = DataFrame(data_frame)

    df_interpolated = df.interpolate(limit_direction="both")
    df_interpolated = df_interpolated.interpolate(limit_direction="both", axis=1)

    return np.array(df_interpolated)
    

def backgroundLevelAnalysis(data_combined, initial_background, make_plots, plot_plots=True,
                           n_iter=20, n_std=15, file_type="flt", image_path=None):
    """
    Iterates background noise, returns the total noise, and corrected data frame.
    
    Runs n_iter iterations of the iterateBackgroundNoise algorithm to find the background
    Estimates the standard deviation from the last n_std iterations. 
    
    Function assumees that the initial_background has already been subtracted from 
    data_combined.
    """
    if file_type=="dr":
        data_combined = interpolateNanValues(data_combined)
        
    iterated_bckgr_level = iterateBackgroundNoise(data_combined, max_iter=n_iter, 
                                                  make_plots=make_plots, plot_plots=plot_plots,
                                                  initial_background=initial_background, 
                                                  n_std=n_std, image_path=image_path)
    
    bckgr_evolution = [np.sum(iterated_bckgr_level[:idx+1]) + initial_background
                         for idx in range(len(iterated_bckgr_level))]
    
    final_background = np.mean(bckgr_evolution[-n_std:])
    background_std = np.std(bckgr_evolution[-n_std:])

    data = data_combined + initial_background - final_background
    
    if make_plots:
        plotTotalNoiseDevelopment(bckgr_evolution, final_background, background_std)
        if image_path != None:
            savefig(image_path + "/2.2_total_noise_development.png")
        if plot_plots:
            show()
    
    return bckgr_evolution, data, final_background, background_std


# -----------------------------------------------------------------------------------------
# In order to make the plots

def addFrameBackgroundModelling(frame, isolist, bckgr_level, popt, plot_number=None):
    """
    Function that makes the plot for the background noise analysis. 
    """
    # x_range in isolist range, scale to fit log-axis
    x_range = np.exp(np.linspace(np.log(min(isolist.sma)), np.log(max(isolist.sma)), 100))
    
    frame.plot(isolist.sma, isolist.intens, "o", label="Galaxy intensities")
    frame.axhline(bckgr_level, c="blue", label=r"$B_i$ = {:.4f}".format(bckgr_level))
    frame.plot(x_range, 10**sersicAndNoiseModel(x_range,*popt), "--",
               label="Sérsic with n={:.2f}".format(popt[1]))
    frame.set_xscale("log")
    frame.set_yscale("log")
    frame.set_xlabel("r [px]", fontsize=13)
    frame.set_ylabel(r"I(r) [$e^- s^{-1} px^{-1}$]", fontsize=13)
    frame.legend(fontsize=10)
    if plot_number != None:
        frame.set_title("Iteration i = {}".format(plot_number))
    return
    

def plotTotalNoiseDevelopment(bckgr_sum, mean_backgr, std_backgr, frame=None):
    """
    Function makes the plot for the development of the noise
    over all iterations
    """
    if frame==None:
        fig = figure(figsize=[8,6])
        frame = fig.add_subplot(111)
    
    x_std = [0,len(bckgr_sum)]
    y_std_max = [mean_backgr + std_backgr, mean_backgr + std_backgr]
    y_std_min = [mean_backgr - std_backgr, mean_backgr - std_backgr]
    
    frame.fill_between(x_std, y_std_max, y_std_min, color="red", alpha=0.5, 
                       label=r"$\sigma_B$ = {:.4f}".format(std_backgr))
    frame.axhline(mean_backgr, c="red", 
                  label=r"$B_{final}$" + " = {:.4f}".format(mean_backgr))
    
    frame.plot(bckgr_sum, label=r"$B_{[i, total]}$")
    frame.set_xlabel("i")
    frame.set_ylabel(r"B [$e^- s^{-1} px^{-1}$]")
    frame.grid()
    frame.set_xlim(0,len(bckgr_sum))   
    frame.legend()
    
    return

############################################################################
# Main background estimation function
############################################################################

def mainBackgroundEstimation(data, mask_cr, image_path=None, make_plots=True, plot_plots=True):
    background = sep.Background(data.astype(np.float32), mask=mask_cr, bw=64, bh=64, fw=3, fh=3)
    total_bckgr = background.globalback
    data -= total_bckgr
    bckgr_evol, data, total_bckgr, std_bckgr = backgroundLevelAnalysis(data, total_bckgr, make_plots, plot_plots=plot_plots, image_path=image_path)
    if len(bckgr_evol) < 3:
        print("Redoing background estimation with lower starting value")
        data += total_bckgr
        total_bckgr = background.globalback - background.globalrms
        data -= total_bckgr
        bckgr_evol, data, total_bckgr, std_bckgr = backgroundLevelAnalysis(data, total_bckgr, make_plots, plot_plots=plot_plots, image_path=image_path)
    if len(bckgr_evol) < 3:
        print("Redoing background estimation with higher starting value")
        data += total_bckgr
        total_bckgr = background.globalback + background.globalrms
        data -= total_bckgr
        bckgr_evol, data, total_bckgr, std_bckgr = backgroundLevelAnalysis(data, total_bckgr, make_plots, plot_plots=plot_plots, image_path=image_path)
    if len(bckgr_evol) < 3:
        print("Background estimation not working, using sextractor value instead")
        data += total_bckgr
        total_bckgr = background.globalback
        data -= total_bckgr
    sex_bckgr = background.globalback
    return data, total_bckgr, sex_bckgr