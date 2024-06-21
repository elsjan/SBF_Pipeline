import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import curve_fit
from scipy.special import erfc


#######################################################################################
# Functions defining the galactic cluster luminosity function and backgounrd galaxy
# luminosity function
#######################################################################################

def fluxToMagnitude(flux, m_0):
    """
    Function that returns the apparent magnitude given the flux of an object
    and a zero point magnitude.
    """
    return -2.5*np.log10(flux) + m_0

def magnitudeToFlux(mag, m_0):
    """
    Function that returns the flux given the apparent magnitude of an object
    and a zero point magnitude.
    """
    return 10**((m_0 - mag)/2.5)

def numberGalacticClusters(m, m_peak, N0_gc, sigma):
    """
    Gaussian Luminosity function representing the galactic 
    cluster luminosiry function
    """
    return (N0_gc/(np.sqrt(2*np.pi)*sigma))*np.exp(-(m-m_peak)**2/(2*sigma**2))

def numberBackgroundGalaxies(m, gamma, N0_bg):
    """
    Power law function representing the background galaxy luminosity function.
    """
    return N0_bg*10**(gamma*m)
    

def combinedLuminosityFunction(m, m_peak, N0_gc, N0_bg, gamma, sigma):
    """
    Combined galactic cluster and background galaxy luminosity function.
    """
    return (numberGalacticClusters(m, m_peak, N0_gc, sigma) +
            numberBackgroundGalaxies(m, gamma, N0_bg))


def combinedLuminosityFunctionLog(m, m_peak, N0_gc, N0_bg, gamma, sigma):
    """
    Logarithm of the combined luminosity function
    """
    return np.log10(combinedLuminosityFunction(m, m_peak, N0_gc, N0_bg, gamma, sigma))

# Vectorise functions for them to be able to handle arrays
combinedLuminosityFunction = np.vectorize(combinedLuminosityFunction)
combinedLuminosityFunctionLog = np.vectorize(combinedLuminosityFunctionLog)


#######################################################################################
# Integrated luminosity functions
#######################################################################################


def globularClusterFluctuations(m_cutoff, m_0, m_peak, N0_gc, sigma):
    """
    Integrated residual power coming from the glovular clusters
    """
    part_1 = 0.5 * N0_gc * 10**(0.8 * (m_0 - m_peak + 0.4 * sigma**2 * np.log(10)) )
    part_2 = erfc((m_cutoff - m_peak + 0.8*sigma**2 * np.log(10))/(np.sqrt(2) * sigma))
    return part_1 * part_2

def backgroundGalaxyFluctuations(m_cutoff, m_0, N0_bg, gamma):
    """
    Integrated resiidual power coming from the background galaxies
    """
    part_1 = N0_bg/((0.8 - gamma)*np.log(10))
    part_2 = 10**(0.8 *(m_0 - m_cutoff) + gamma*m_cutoff)
    return part_1 * part_2

def totalResidualPower(m_0, m_cutoff, m_peak, N0_gc, N0_bg, gamma, sigma):
    """
    Returns the combined residual power of the gc luminosity function 
    and the bg luminosity funcion
    """
    sig2_gc = globularClusterFluctuations(m_cutoff, m_0, m_peak, N0_gc, sigma)
    sig2_bg = backgroundGalaxyFluctuations(m_cutoff, m_0, N0_bg, gamma)
    return sig2_gc + sig2_bg



#######################################################################################
# Functions required for the fit and uncertainties
#######################################################################################


def createLuminosityHistogram(flux, m_0, n_pix, bin_width=0.4):
    """
    Convert fluxes to luminosity and return the binned values
    """
    mags = fluxToMagnitude(flux, m_0)
    
    bin_edges = np.arange(int(min(mags)), max(mags)+bin_width, bin_width)
    bin_centers = [np.sum(bin_edges[i:i+2])/2 for i in range(len(bin_edges)-1)]
    
    hist, _ = np.histogram(mags, bin_edges)
    return bin_centers, hist/bin_width/n_pix
    

def fitLogLuminosityFunction(m, N_tot, gamma, sigma, n_pix=1):
    """
    Fit of the galactic cluster and background galaxy luminosity function
    """
    
    LfToFit = partial(combinedLuminosityFunctionLog, gamma=gamma, sigma=sigma)
    
    fit_params, _ = curve_fit(f=LfToFit, xdata=m, ydata=N_tot,
                       p0=[m[np.where(N_tot==max(N_tot))[0][0]], 
                           len(N_tot)*0.6,  1e-10], 
                           bounds=[[min(m), 0, 0],[max(m)+1e-5, np.inf, np.inf]])   
    
    return fit_params


def findCompletenessLimit(mag, N_tot, parameters, completeness_threshold=0.8, 
                          gamma=0.25, sigma=1.2):
    """
    Find the completeness limit. 
    Finds the values for which the histogram values are above the fitted 
    luminosity threshold, and finds the maximum value that falls within that 
    bound. The bin corner of that bin is determined the completeness limit
    """
    idx_include = (np.log10(N_tot) > (combinedLuminosityFunctionLog(mag, 
                                      *parameters, gamma, sigma) - 
                                      (1-completeness_threshold) * np.abs(
                                      combinedLuminosityFunctionLog(mag, 
                                      *parameters, gamma, sigma))))
    
    if np.any(idx_include) == False:
        idx_limit = 0
    else:
        idx_limit = np.max(np.where(idx_include))
    
    completeness_limit = np.mean(mag[idx_limit:idx_limit+2])
    return completeness_limit
    

def includeToConstraints(max_fit_included_value, completeness_limit, 
                         bool_constr=None):
    """
    Function that controlls whether a fitted luminosity function gets indicated
    as sufficient quality to be included to the final residual power estimate
    
    The conditions are:
    - The values included in the fit all are within the completeness limit.
    - The max value included in the fit is within 0.6 mag from the completeness 
      limit
    """
    condition_1 = max_fit_included_value < completeness_limit
    condition_2 = abs(max_fit_included_value - completeness_limit) < 0.6
    if bool_constr != None:
        condition_1 = True
    return condition_1 & condition_2

    
def iterateToOptimalResidualPower(flux, m_0, n_pix, gamma, sigma, plot=True, 
                                 image_path=None, bool_constr=None):
    """
    Function iterates to the optimal residual power. 
    
    Funcion iterates over a number of bin widths and number of included data
    points. The combined Luminosity function is fitted and the parameters are 
    included depending on the defined conditions
    """
    
    bin_widths = [0.4, 0.4, 0.3, 0.2]
    params = []
    bool_include = []
    completeness_limit = []
    
    # for plotting
    if plot == True:
        fig, subplots = plt.subplots(2,2, figsize=[8,8])
        subplots = subplots.flatten()
        idx_plot = 0
        idx_bin_width = 0
    
    for bin_size in bin_widths:
        mag, N_tot = createLuminosityHistogram(flux, m_0, n_pix=1)
        
        # iterate over bin sizes:
        for i in range(int(0.6/bin_size),int(2/bin_size)): 
            idx_include = np.array(N_tot) != 0 
            idx_fit = len(np.array(N_tot)[idx_include]) - i - 1
            
            mag_i = np.array(mag)[idx_include][1:idx_fit]
            N_tot_i = np.array(N_tot)[idx_include][1:idx_fit]
            
            params_i = fitLogLuminosityFunction(mag_i, np.log10(N_tot_i),
                                                gamma, sigma, n_pix=1)
            params.append(params_i)
            
            comp_lim_i = findCompletenessLimit(np.array(mag)[idx_include], 
                                               np.array(N_tot)[idx_include], 
                                               params_i, 
                                               gamma=gamma, sigma=sigma)
            
            completeness_limit.append(comp_lim_i)
            bool_include.append(includeToConstraints(np.max(mag_i), comp_lim_i, 
                                                     bool_constr)) 
            
        if plot == True:
            addFrameLuminosityFunction(subplots[idx_plot], bin_size, 
                                       mag, N_tot,
                                       params[idx_bin_width:],
                                       bool_include[idx_bin_width:],
                                       sigma, gamma, n_pix)
            
            idx_plot += 1
            idx_bin_width = len(params)
    
    if plot == True:
        fig.tight_layout()
        if image_path != None:
            plt.savefig(image_path + "/6.1_binned_luminosity_functions.png")
        plt.show()
            
    return (np.array(params)[bool_include], 
             np.mean(np.array(completeness_limit)[bool_include]))


def findResidualPower(m_0, m_cutoff, params, gamma, sigma,
                     plot=True, image_path=None, flux=None, n_pix=None):
    """
    Find residual power from a list of params. Plot if necessary.
    """
    res_power = []
    for param in params:
        res_power_i = totalResidualPower(m_0, m_cutoff, *param, gamma, sigma)
        res_power.append(res_power_i)
    
    if plot == True:
        plotFinalPowerEstimate(flux, m_0, m_cutoff, params, gamma, sigma, 
                               image_path, n_pix=n_pix)
    
    return np.mean(res_power)/n_pix, np.std(res_power)/n_pix
    

def getSourceMaskAboveCutoff(fluxes, segmap, m_0, m_cutoff):
    """
    Identifies and masks the sources in the segmentation map that are below the
    cutoff magnitude.
    """
    mask_all = segmap != 0
    mask_above_thr = np.zeros(segmap.shape, dtype=bool)
    
    idx_above_threshold = np.where(fluxes < magnitudeToFlux(m_cutoff, m_0))[0]
    for idx in idx_above_threshold:
        mask_above_thr = mask_above_thr | (segmap == idx + 1)
        
    final_mask = mask_all & ~mask_above_thr
    return final_mask
    


    
#######################################################################################
# Main function
#######################################################################################
    
def findMaskAndResidualPower(fluxes, segmap, n_pix, obs_filter, plot=True, 
                      image_path=None):
    """
    fits the background galaxy and globular cluster luminosity function.
    
    Uses this to estimate the residual power from unmasked background galaxies
    and globular clusters. Also the final source mask is returned.
    """
    
    m_0, gamma, sigma = getFilterDependentValues(obs_filter)
    
    params, m_cutoff = iterateToOptimalResidualPower(fluxes, m_0, 
                                              n_pix, gamma, sigma, plot=plot, 
                                              image_path=image_path)
    
    if len(params) == 0:
        params, m_cutoff = iterateToOptimalResidualPower(fluxes, m_0, 
                                              n_pix, gamma, sigma, plot=plot, 
                                              image_path=image_path, 
                                              bool_constr=True)
        
    res_power, sig_res_power = findResidualPower(m_0, m_cutoff, params, 
                                                 gamma, sigma, plot=plot,
                                                 image_path=image_path,
                                                 flux=fluxes, n_pix=n_pix)
    
    source_mask = getSourceMaskAboveCutoff(fluxes, segmap, m_0, m_cutoff)
    
    return source_mask, res_power, sig_res_power


#######################################################################################
# Function storing zero-points
#######################################################################################

def getFilterDependentValues(obs_filter):
    """
    Function returns the 0-point magnitude corresponding to the given HST filter band.
    Also the background galaxy luminosity function slope is returned as well as the 
    sigma on the galactic cluster luminosity function.
    
    Sources:
    - Sirianni et al. (2005): m0 for F475W, F850LP
      https://iopscience.iop.org/article/10.1086/444553/pdf
    - Jensen et al. (2015): m0, gamma, sigma for F160W, F110W
      https://arxiv.org/pdf/1505.00400
    - Mei et al. (2005): gamma, sigma for F850LP
      https://iopscience.iop.org/article/10.1086/429554/pdf
    - Benitez et al. (2004): gamma for F475W
      https://iopscience.iop.org/article/10.1086/380120/pdf
    """
    if obs_filter == "F110W":
        return 26.8223, 0.25, 1.2
    elif obs_filter == "F160W":
        return 25.9463, 0.25, 1.2
    elif obs_filter == "F475W":
        return 26.068, 0.32, 1.2
    elif obs_filter == "F850LP":
        return 24.862, 0.35, 1.2
        

       #######################################################################################
# For plotting
#######################################################################################

    
    
def addFrameLuminosityFunction(frame, bin_width, mag, N_tot, params, 
                               bool_include, sigma, gamma, n_pix=1):
    """
    Add frame with luminosity function plots
    """
    params = np.array(params)
    
    x_range = np.linspace(np.min(mag)-0.5, np.max(mag)+0.5, 100)
    
    frame.plot(mag, N_tot/n_pix, "o", c="green", label = "Data points")
    bool_include = np.array(bool_include, dtype=bool)
    
    
    for idx in range(len(params[~bool_include])):
        if idx == 0:
            frame.plot(x_range, combinedLuminosityFunction(x_range, 
                       *params[~bool_include][idx], gamma, sigma)/n_pix, 
                       c="grey", alpha =0.5, label = "Not included")
        else:
            frame.plot(x_range, combinedLuminosityFunction(x_range, 
                       *params[~bool_include][idx], gamma, sigma)/n_pix, 
                       c="grey", alpha =0.5)
    
    for idx in range(len(params[bool_include])):
        if idx == 0:
            frame.plot(x_range, combinedLuminosityFunction(x_range, 
                       *params[bool_include][idx], gamma, sigma)/n_pix, 
                       c="black", label = "Included")
        else:
            frame.plot(x_range, combinedLuminosityFunction(x_range, 
                       *params[bool_include][idx], gamma, sigma)/n_pix, 
                       c="black")
    
    frame.legend()
    
    frame.set_yscale("log")
    frame.set_ylabel(r"$N_{obj} mag^{-1} pix^{-1}$")
    frame.set_xlabel("mag")
    
    frame.set_xlim(min(x_range), max(x_range))
    frame.set_title("bin widh = {:.1f} mag".format(bin_width))
    
    return


def plotFinalPowerEstimate(flux, m_0, m_cutoff, params, gamma, sigma, 
                            image_path=None, n_pix=1):
    """
    Make a plot of the residual power estimate
    """
    fig = plt.figure(figsize=[6,6])
    frame = fig.add_subplot(111)
    
    x_range = np.arange(18, 30, 100)
    if np.any(flux != None):
        mag, N_tot = createLuminosityHistogram(flux, m_0, n_pix=1)
        mag, N_tot = mag[1:], N_tot[1:]
        plt.plot(mag, N_tot/n_pix, "o", color="green", label="observed points")
        x_range = np.linspace(np.min(mag)-0.5, np.max(mag)+0.5, 100)
    
    mean_params =  np.mean(params, axis=0)
    frame.plot(x_range, combinedLuminosityFunction(x_range, 
                *mean_params, gamma, sigma)/n_pix, c="black", 
               label="combined luminosity function")
    
    frame.plot(x_range, numberBackgroundGalaxies(x_range, gamma, 
                                    mean_params[2])/n_pix,
               "--", c="black", label="background galaxy component")
    frame.plot(x_range, numberGalacticClusters(x_range, mean_params[0], 
                                    mean_params[1], sigma)/n_pix,
               ":", c="black", label="galactic cluster component")
    
    frame.axvspan(m_cutoff, np.max(x_range), alpha=0.5, color='red', 
                  label="completeness limit")
    
    frame.legend()
    
    frame.set_yscale("log")
    frame.set_ylabel(r"$N_{obj} mag^{-1} pix^{-1}$")
    frame.set_xlabel("mag")
    
    frame.set_xlim(min(x_range), max(x_range))
    
    frame.set_ylim(numberBackgroundGalaxies(x_range[0], gamma, 
                                    mean_params[2])/n_pix)
    if image_path != None:
        plt.savefig(image_path + "/6.2_final_gc_bg_luminosity_function.png")
    
    plt.show()
    return
