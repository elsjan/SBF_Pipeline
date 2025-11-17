############
# Write intro
###########

import numpy as np

def computeColors(data, mask_combined, mzp=30.132, scale=0.1, gain=86.95309785869, filter=None,EBV=0):
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

