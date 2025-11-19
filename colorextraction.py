############
# Write intro
###########

import numpy as np

def downscale_mask_majority(mask, factor=3):
    # reshape into blocks of (factor x factor)
    h, w = mask.shape
    new_h = h // factor
    new_w = w // factor
    
    # reshape to: (new_h, factor, new_w, factor)
    blocks = mask.reshape(new_h, factor, new_w, factor)
    
    # count number of True in each block
    true_count = blocks.sum(axis=(1, 3))
    
    # majority: True only if strictly more than half
    # for 3×3 blocks, majority means >=5 Trues
    return true_count >= ((factor * factor) // 2 + 1)

def trim(data, p):
    print(p)
    p = int(p)//2
    
    data_new = data[p:-p,p:-p]
    return data_new

def computeSurfaceBrightness(data, mask_combined, mzp=30.132, scale=0.1, gain=86.95309785869, filter=None,EBV=0):
    # expecting background subracted data
    # standard mzp, scale, and gain set to Euclid VIS (LSB)
    # scale: arcsec per pixel
    # gain: e- per ADU
    # unsure on what exactly went into the calculation of mzp --> read the Euclid pipeline paper
    # to use EBV we would also need R (extinction coefficients?), not applied at this moment, since expected to give little change
    if filter != None:
        if filter == "VIS":
            mzp=30.132
            scale=0.1
            gain = 86.95309785869
        elif filter == "H":
            mzp=30.00
            scale=0.3
            gain = 5.451803290107
        else:
            print("unkown filter: ", filter)

    data_us = data*mask_combined
    adu_per_pix = data_us.sum()/len(data_us)
    adu_per_arcsec = adu_per_pix/(scale**2)
    e_per_arcsec = adu_per_arcsec*gain
    sb = mzp - 2.5 * np.log10(e_per_arcsec)
    return sb

def computeColor(dataB, dataR, mask, do_print=False):
    mask = ~mask.astype(bool)
    if dataB.shape[0] < dataR.shape[0]*3:
        newshape = int(np.floor(dataB.shape[0]/6))
        dataB = trim(dataB, dataB.shape[0]-newshape*6)
        dataR = trim(dataR, dataR.shape[0]-newshape*2)
        mask = trim(mask, mask.shape[0]-newshape*6)
        print(dataB.shape[0])
        print(dataR.shape[0])
        print(mask.shape[0])
    if dataB.shape[0] > dataR.shape[0]*3:
        newshape = int(np.floor(dataR.shape[0]))
        dataB = trim(dataB, dataB.shape[0]-newshape*3)
        dataR = trim(dataR, dataR.shape[0]-newshape)
        mask = trim(mask, mask.shape[0]-newshape*3)
        print(dataB.shape[0])
        print(dataR.shape[0])
        print(mask.shape[0])
    mask_smaller = downscale_mask_majority(mask)
    dataBmasked = np.ma.masked_array(dataB, mask)
    dataRmasked = np.ma.masked_array(dataR, mask_smaller)
    countsB = np.nansum(dataBmasked)
    countsR = np.nansum(dataRmasked)
    magB = 30.132-2.5*np.log10(countsB)
    magR = 30-2.5*np.log10(countsR)
    if do_print:
        print("magB, magI, amountpixelsB, amountpixelsR, scaledamountpixelsB")
        print(magB, magR, len(dataBmasked), len(dataRmasked), len(dataRmasked)/9)
        print()
    return magB-magR

def computeMagnArea(data, mask, filter='VIS', do_print=False):
    if filter == "VIS":
        mzp=30.132
        scale = 0.1
    elif filter == "H":
        mzp=30.00
        scale = 0.3
        mask = downscale_mask_majority(mask)
    else:
        print("unkown filter: ", filter)
    mask = ~mask.astype(bool)
    masked_data = np.ma.masked_array(data, mask)
    counts = np.nansum(masked_data)
    
    area = (len(masked_data.flatten()) - np.isnan(masked_data).sum())*(scale**2)
    print(area)
    mag = mzp -2.5*np.log10(counts)
    mag_arcsec = mzp -2.5*np.log10(counts/area)
    if do_print:
        print("mag")
        print(mag)
    return mag_arcsec, mag
