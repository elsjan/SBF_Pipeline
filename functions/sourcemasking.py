import numpy as np
import sep
import matplotlib.pyplot as plt
from mgefit.find_galaxy import find_galaxy
from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture


# own functions
from residualpowerestimator import findMaskAndResidualPower
from plotting import imdisplay

def dataCutout(data, x0, y0, frame=250):
    """
    Make a cutout for plotting.
    """ 
    idx_x1 = int(x0) - frame
    idx_x2 = int(x0) + frame    
    
    idx_y1 = int(y0) - frame    
    idx_y2 = int(y0) + frame    
    
    return data[idx_x1:idx_x2, idx_y1:idx_y2]


def unmaskMaxArea(objects, segmap, max_area):
    """
    Function unmasks the sources that have an area larger than max_area pixels.
    These are removed from the segmap and set to 0
    """
    indices_removed = []
    for idx in np.arange(1,np.max(segmap)):
        if len(segmap[segmap == idx]) > max_area:
            segmap[segmap == idx] = 0
            indices_removed.append(idx-1)
            
    objects = np.delete(objects, indices_removed, 0)
    return objects, segmap


def findCenterSourceMask(segmap, x0, y0, pix_from_center=10):
    """
    Return the mask of all the "objects" that are located within
    pix_from_center pixels from the center.
    """
    mask = np.zeros(segmap.shape, dtype=bool)
    
    # Obtain indices of segmap in the center
    segment_values = set(segmap[x0-pix_from_center:x0+pix_from_center,
                                y0-pix_from_center:y0+pix_from_center].flatten())
    
    for value in segment_values:
        if value != 0:
            mask[segmap==value] = 1
    
    return mask

# Same as in empiricalpsf
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


##############################################################################################
# SEXtractor functions
##############################################################################################


def obtainSextractorSources(image, mask_0, thr, 
                            box_width=64, subtract_sep_bckgr=True,
                            min_area=5, max_area=None, 
                            scale_to_galmodel=False, img_model=None,
                            deblend_nthresh=32, deblend_contrast=0.005):
    """
    Function that returns a list of objects and a segmentation map
    for a given image and sextractor (sep) parameters
    """
    # First determine backgr following Sextractor, then find sources
    try:
        sep_background = sep.Background(image, mask=mask_0, 
                                        bw=box_width, bh=box_width)
    except: # sometimes byte conversion is necessary
        sep_background = sep.Background(image.astype(image.dtype.newbyteorder('=')), 
                                        mask=mask_0, bw=box_width, bh=box_width)
    
    img_subtracted = np.copy(image)
    if subtract_sep_bckgr==True:
        img_subtracted = image - sep_background.back()        
    
    # If not scaled to galmodel, uncertainty is the globalrms; a constant value
    uncertainty = sep_background.globalrms
    
    if scale_to_galmodel==True: # Else, it is the sep_background + scaled image model
        uncertainty = sep_background.back() + img_model**0.5
        
    # Again, sometimes byte conversion is necessary
    try:
        objects, segmap = sep.extract(img_subtracted, thr, mask=mask_0, 
                                  err=uncertainty,
                                  minarea=min_area,
                                  segmentation_map=True,
                                  deblend_nthresh=deblend_nthresh, 
                                  deblend_cont=deblend_contrast)
    except:
        img_subtracted = img_subtracted.byteswap().newbyteorder()
        objects, segmap = sep.extract(img_subtracted, 
                                  thr, mask=mask_0.byteswap().newbyteorder(), 
                                  err=uncertainty,
                                  minarea=min_area,
                                  segmentation_map=True,
                                  deblend_nthresh=deblend_nthresh, 
                                  deblend_cont=deblend_contrast)
    
    if max_area != None:
        objects, segmap = unmaskMaxArea(objects, segmap, max_area)
    
    return objects, segmap


def extractSextractorMask(image, mask_0, img_model, thr, 
                          center_x0=0, center_y0=0,
                           return_objects=False, box_width=64,
                           subtract_sep_bckgr=True,
                           min_area=5, max_area=None,
                           scale_to_galmodel=False,
                           deblend_nthresh=32, deblend_contrast=0.005):
    """
    Function identifies sources in an image using SEXtractor. A mask is 
    then created from the segmentation map.
    
    The sources in the center are also detected and returned.
    
    Depending on return_obj, the objects are returned, otherwise the 
    centermask is returned. 
    """
    
    obj, segmap = obtainSextractorSources(image, mask_0, thr, 
                      box_width, subtract_sep_bckgr,
                      min_area, max_area, scale_to_galmodel, img_model,
                      deblend_nthresh, deblend_contrast)
    
    mask = segmap != 0
    
    if return_objects == True:
        return segmap, obj
    else:
        centermask = findCenterSourceMask(segmap, center_x0, center_y0)
        return mask, centermask


def findMaskAndCenterMask(image, mask_0, img_model, thr, 
                       inv_objects=True, box_width=64,
                       subtract_sep_bckgr=True,
                       min_area=5, max_area=None,
                       scale_to_galmodel=False,
                       deblend_nthresh=32, deblend_contrast=0.005):
    """
    Find the sourcemask for a Sextractor run.
    If inv_objects is True, SEXtractor is also run for the inverse image. 
    """
    f = find_galaxy(img_model, plot=False, quiet=True)
    
    mask, centermask = extractSextractorMask(image, mask_0, img_model, thr, 
                        f.xpeak, f.ypeak, False, 
                        box_width, subtract_sep_bckgr, min_area, max_area,
                        scale_to_galmodel, deblend_nthresh, deblend_contrast)
    
    if inv_objects:
        mask_inv, centermask_inv = extractSextractorMask(-image, mask_0, img_model, thr, 
                        f.xpeak, f.ypeak, False, 
                        box_width, subtract_sep_bckgr, min_area, max_area,
                        scale_to_galmodel, deblend_nthresh, deblend_contrast)
        
        mask = mask | mask_inv
        centermask = centermask | centermask_inv
        
    return mask, centermask

def findMaskAndSepObjects(image, mask_0, img_model, thr, 
                       inv_objects=True, box_width=64,
                       subtract_sep_bckgr=True,
                       min_area=5, max_area=None,
                       scale_to_galmodel=False,
                       deblend_nthresh=32, deblend_contrast=0.005):
    """
    Find the sourcemask for a Sextractor run.
    If inv_objects is True, SEXtractor is also run for the inverse image.
    The mask and object list are returned.
    
    only the objects are returned, not the inv_objects
    """ 
    segmap, objects = extractSextractorMask(image, mask_0, img_model, thr, 0, 0, True,
                                  box_width, subtract_sep_bckgr, min_area, max_area,
                                  scale_to_galmodel, deblend_nthresh, deblend_contrast)
    
    if inv_objects:
        f = find_galaxy(img_model, plot=False, quiet=True)
        mask_inv, center_mask = extractSextractorMask(-image, mask_0, img_model, 
                                  thr, f.xpeak, f.ypeak, False, 
                                  box_width, subtract_sep_bckgr, min_area, max_area,
                                  scale_to_galmodel, deblend_nthresh, deblend_contrast)
        mask = segmap != 0
        mask = mask | mask_inv
        return mask, objects
    
    else:
        return segmap, objects
    

def centralAnnulusMask(img_model, geometry=None, inner_radius=50, inner_percentage=None):
    """
    function masks a circular annulus of a given radius arounc the center
    of a galaxy.
    """
    img_model = img_model.copy()
    img_model = np.where(np.isnan(img_model), np.nanmedian(img_model), img_model)
    if geometry == None:
        f = find_galaxy(img_model, quiet=True, plot=False)
        if f.pa < 90:
            geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                                sma=f.majoraxis, eps=f.eps, 
                                pa=(f.pa+90)*np.pi/180, astep=0.1)
        else:
            geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                                sma=f.majoraxis, eps=f.eps, 
                                pa=(f.pa-90)*np.pi/180, astep=0.1) 
    if inner_percentage != None:               # added this to make galaxy size dependent option
        inner_radius = geometry.sma*inner_percentage
    mask = maskCircle(img_model, geometry.x0, geometry.y0, rout=inner_radius, rin=0)
    return mask
    
##############################################################################################
# MAIN functions
##############################################################################################

def maskBackgroundSources(data, mask_cr=None, make_plots=False, plot_plots=False, detect_thresh=3, minarea=7, maxarea=None, r=2.5, image_path=None, final=False, original_image=None):
    """
    Detect and mask background sources using SEP (SExtractor).
    Works with masked arrays or normal numpy arrays.
    """

    # Estimate and subtract background
    bkg = sep.Background(data.astype(np.float32), mask=mask_cr, bw=64, bh=64, fw=3, fh=3)
    # data_sub = data - bkg.back()

    # Detect sources
    objects, segmap = sep.extract(data, thresh=detect_thresh * bkg.globalrms,
                          mask=mask_cr, minarea=minarea, segmentation_map=True)
    if maxarea != None:
        objects, segmap = unmaskMaxArea(objects, segmap, maxarea)

    # Mask them
    mask_sources = np.zeros(data.shape, dtype=bool)
    sep.mask_ellipse(mask_sources, objects['x'], objects['y'],
                     objects['a'], objects['b'], objects['theta'], r=r)

    # Combine
    mask_combined = mask_sources | mask_cr

    if make_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(data, ax, percentlow=1, percenthigh=99, scale='asinh')
        plt.title("Detected Sources Mask")
        plt.imshow(mask_combined, origin='lower', cmap='Reds', alpha=0.5)
        if image_path != None:
            image_title = "6.1_source_mask.png"
            plt.savefig(image_path + "/" + image_title)
        if plot_plots:
            plt.show()

    
    return mask_combined

def createRequiredVariables(data, model_final, source_mask_final, total_background, geometry, make_plots=False, plot_plots=False, image_path=None):
    """
    From the data and the model, the nri, model mask, and total mask is returned.
    """

    # geometry.sma *= 3
    aperture = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1-geometry.eps), geometry.pa)
    aperture_mask_obj = aperture.to_mask(method='center',subpixels=1)
    aperture_mask = aperture_mask_obj.to_image(data.shape).astype(bool)

    inner_radius = geometry.sma*0.15

    mask_center = maskCircle(data, geometry.x0, geometry.y0, rout=inner_radius, rin=0)


    # mask_model = model_final <= 1.5 #* total_background   #ballsy change
    # mask_combined = np.array(~(mask_model | source_mask_final), dtype=int)

    mask_combined = ~(mask_center | source_mask_final)
    mask_combined &= aperture_mask

    nri = (data - model_final) / np.sqrt(model_final)
    nri[~np.isfinite(nri)] = 0

    nri *= mask_combined

    if make_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(nri, ax, percentlow=1, percenthigh=99, scale='linear')
        plt.title("NRI")
        if image_path != None:
            image_title = "7.1_nri.png"
            plt.savefig(image_path + "/" + image_title)
        if plot_plots:
            plt.show()
    return aperture_mask, mask_combined, nri


##############################################################################################
# Unused old functions
# Maybe double check to make sure
##############################################################################################


def obtainSepParameters(obs_filter, data_type):
    """
    The SEP parameters required for a given data and filter input are returned
    
    This function determines how the sources are exactly masked.
    
    The explanation is more or less as follows:
    The length of the returned parameter list determines the number of runs.
    
    the indexes correspond to the following:
    idx0: threshold          :float: Relative threshold for detection
    idx1: inv_objects        :bool:  Must negative sources be identified
    idx2: box_width          :int:   For finding sep background
    idx3: subtract_sep_bckgr :bool:  Must the sep background be subtracted before
                                     identifying sources
    idx4: min_area           :int:   Minimum area for a source to be detected.
    idx5: max_area           :int:   Maximum area for a source to be detected.
    idx6: scale_to_galmodel  :bool:  Must the backround be scaled to the image model
    """
    if (obs_filter == "F110W") & (data_type =="dr"):
        return [[0.5, True,  256, True, 50,  None, True ],
                [0.5, False, 64,  True, 5,   None, True ]]    
    elif (obs_filter == "F160W") & (data_type == "flt"):
        return [[0.5, True,  256, True, 50,  None, True ],
                [5,   True,  256, True, 1,   4,    False],
                [0.5, False, 64,  True, 5,   None, True ]]
    elif (obs_filter == "F160W") & (data_type =="dr"):
        return [[0.5, True,  256, True, 50,  None, True ],
                [0.5, False, 64,  True, 5,   None, True ]]     

    elif (obs_filter == "F475W") & (data_type =="dr"):
        return [[0.1, True,  256, True, 100, None, True ],
                [0.1, False, 64,  True, 5,   None, True ]]
    elif (obs_filter == "F850LP") & (data_type =="dr"):
        return [[0.2, True,  256, True, 100, None, True ],
                [0.2, False, 64,  True, 5,   None, True ]]
    elif (obs_filter == "VIS") & (data_type =="flt"):
        return [[3, True,  64, True, 7, None, False ],
                [3, False, 64,  True, 7, None, False ]]
    else:
        print("No obs_filter, data_type pair matched. Default SEP parameters used.")
        return [[0.2, True,  256, True, 100, None, True ],
                [0.2, False, 64,  True, 5,   None, True ]]
        
##############################################################################################        
# Initial source mask
##############################################################################################

def findInitialSourceMask(residual_image, img_model, obs_filter, data_type, mask_0=None, 
                          plot=False, image_path=None, image_title="4.1_initial_nri.png"):
    """
    Makes an estimate of the source mask given the residual image and the img model. 
    
    Sources are only identified in the area where img_model != 0
    The combined mask and centermask of each run are both returned.
    """
    if np.any(mask_0 == None):
        mask_0 = img_model == 0
    else: 
        mask_0 = mask_0 | (img_model == 0)
        
    parameters = obtainSepParameters(obs_filter, data_type)
    
    mask = np.zeros(residual_image.shape, dtype=bool)
    centermask = np.zeros(residual_image.shape, dtype=bool)

    for params in parameters:
        mask_i, centermask_i = findMaskAndCenterMask(residual_image, mask_0, 
                                                     img_model, *params)
        mask = mask | mask_i
        centermask = centermask | centermask_i
        mask_0 = mask_0 | mask
        
    if plot==True:
        plotMaskedNri(img_model, residual_image, mask_0, image_path, image_title)
        
    return mask, centermask

##############################################################################################        
# Final source mask
##############################################################################################

def findFinalSourceMaskInitial(residual_image, img_model, obs_filter, 
                               data_type, mask_0=None, total_background=0,
                               gal_bckgr_fraction = 0.6):
    """
    Finds the part of the source mask that contains the largest fluctuations,
    i.e. each component that is not the final run.
    
    The model mask is generated by looking at the area for which the galaxy
    model is at least 60% of the total flux (i.e. model + backgorund level)
    """
    galaxy_factor = gal_bckgr_fraction/(1-gal_bckgr_fraction)
    if np.any(mask_0 == None):
        mask_0 = img_model <= galaxy_factor * total_background
    else: 
        mask_0 = mask_0 | (img_model <= galaxy_factor * total_background)
    
    parameters = obtainSepParameters(obs_filter, data_type)
    
    mask = np.zeros(residual_image.shape, dtype=bool)
    for params in parameters[:-1]:
        mask_i, _ = findMaskAndSepObjects(residual_image, mask_0, 
                                                     img_model, *params)
        mask = mask | mask_i
        mask_0 = mask_0 | mask
    
    return mask_0
    

def findSourcesResidualPower(residual_image, img_model, obs_filter, 
                             data_type, mask_0=None, last_threshold=None):
    """
    Function that finds the sources that make up the residual power.
    
    The mask, segmentation map and the object list are returned
    """
    if np.any(mask_0 == None):
        mask_0 = img_model == 0
        
    parameters = obtainSepParameters(obs_filter, data_type)
    if np.any(last_threshold != None):
        parameters[-1][0] = last_threshold

    segmap, objects = findMaskAndSepObjects(residual_image, mask_0, 
                                            img_model, *parameters[-1])
    
    return segmap, objects["flux"]
    

def findFinalSourceMask(residual_image, img_model, obs_filter, 
                        data_type, mask_0=None, total_background=0,
                        final_threshold=None, gal_bckgr_fraction=0.6,
                        central_mask_radius=50,
                        plot=False, image_path=None):
    """
    Function first identifies bright fore and background sources.
    
    Then the segmentation map of the final sources is identified,
    residual power is estimated, and the final source mask is returned
    """
    mask_foreground = findFinalSourceMaskInitial(residual_image, img_model, 
                        obs_filter, data_type, mask_0, total_background, 
                        gal_bckgr_fraction=gal_bckgr_fraction)
    
    segmap, flux = findSourcesResidualPower(residual_image, img_model, 
                        obs_filter, data_type, mask_0=mask_foreground, 
                        last_threshold=final_threshold)
    
    n_pix = len(mask_foreground[mask_foreground==0])

    
    mask_residual, res_power, sig_res_power = findMaskAndResidualPower(flux, 
                                               segmap, n_pix, obs_filter, 
                                               plot=plot, image_path=image_path)
    
    #Normalise with galaxy flux
    res_power     /= np.mean(img_model[~mask_foreground])
    sig_res_power /= np.mean(img_model[~mask_foreground])
    
    final_mask = mask_foreground | mask_residual
    
    final_mask = final_mask | centralAnnulusMask(img_model, inner_radius=central_mask_radius)

    if plot==True:
        plotMaskedNri(img_model, residual_image, final_mask, image_path, 
                      image_title="6.3_final_nri.png")
        
    return final_mask, res_power, sig_res_power
    

        
    
##############################################################################################
# For plotting
##############################################################################################

def plotMaskedNri(img_model, res_image, total_mask, image_path, image_title,
                 plot_cutout=False, cutout_size=300):
    """
    Makes a plot showing the whole NRI, with all masked pixels set to 0.
    """
    img_model_altered = np.copy(img_model)
    img_model_altered[img_model==0] = 1
    
    nri = res_image/np.sqrt(img_model_altered)

    mean = np.mean(nri[~total_mask]) 
    std  = np.std((nri[~total_mask]))

    fig = plt.figure(figsize=[16,16])
        
    frame = fig.add_subplot(111)
    
    nri[total_mask] = 0
    
    if plot_cutout==True:
        f = find_galaxy(img_model, plot=False, quiet=True)
        nri = dataCutout(nri, f.xpeak, f.ypeak, cutout_size)

    frame.imshow(nri, cmap="gray", vmin=mean-7*std, vmax=mean+7*std) #, origin=1)
    
    if image_path != None:
        plt.savefig(image_path + "/" + image_title)
    
    plt.show()
    return

