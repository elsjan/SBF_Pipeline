import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u

dwarfs_cat = pd.read_csv("Dorado_cat_xin/dorado_t21_t24_mer.csv")

# dwarfs_cat['RIGHT_ASCENSION'], dwarfs_cat['DECLINATION'] OBJECT_ID, Re_arcsec, mag, ColorID




hdul= fits.open('ERO-Fornax/Euclid-VIS-ERO-Fornax-LSB.DR3.fits')
full_data = hdul[0].data
full_header = hdul[0].header
wcs = WCS(full_header)
hdul.close()
for name in FCCTable['Name']:
    cutout_size = round(FCCTable[FCCTable['Name']==name]['Re'].value[0]*30)*3
    ra_center = FCCTable[FCCTable['Name']==name]['ra']
    dec_center = FCCTable[FCCTable['Name']==name]['dec']
    position = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg), frame='icrs')
    cutout = Cutout2D(full_data, position, (cutout_size, cutout_size), wcs=wcs, copy=True)
    new_header = full_header.copy()
    new_header.update(cutout.wcs.to_header())
    hdu_cutout = fits.PrimaryHDU(data=cutout.data, header=new_header)
    try:
        os.makedirs("FornaxLSB/{}/flt".format(name))
    except FileExistsError:
        # directory already exists
        pass
    hdu_cutout.writeto("FornaxLSB/{}/flt/{}.fits".format(name, name), overwrite=True)

    fieldcutout = Cutout2D(full_data, position, (cutout_size*2, cutout_size*2), wcs=wcs, copy=True)
    new_header_field = full_header.copy()
    new_header_field.update(fieldcutout.wcs.to_header())

    hdu_fieldcutout = fits.PrimaryHDU(data=fieldcutout.data, header=new_header_field)
    hdu_fieldcutout.data[cutout_size//2:cutout_size*3//2, cutout_size//2:cutout_size*3//2] = np.nan ## FIXXXX
    hdu_fieldcutout.writeto("FornaxLSB/{}/field_{}.fits".format(name, name), overwrite=True)
    

    plt.figure()
    plt.imshow(cutout.data, origin='lower', cmap='gray', norm='log')
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.title('FITS Cutout {}'.format(name))
    plt.show()


    # now the same in the H-band

hdul= fits.open('ERO-Fornax/Euclid-NISP-H-ERO-Fornax-LSB.DR3.fits')
full_data = hdul[0].data
full_header = hdul[0].header
wcs = WCS(full_header)
hdul.close()
for name in FCCTable['Name']:
    cutout_size = round(FCCTable[FCCTable['Name']==name]['Re'].value[0]*30)
    ra_center = FCCTable[FCCTable['Name']==name]['ra']
    dec_center = FCCTable[FCCTable['Name']==name]['dec']
    position = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg), frame='icrs')
    cutout = Cutout2D(full_data, position, (cutout_size, cutout_size), wcs=wcs, copy=True)
    new_header = full_header.copy()
    new_header.update(cutout.wcs.to_header())
    hdu_cutout = fits.PrimaryHDU(data=cutout.data, header=new_header)
    try:
        os.makedirs("FornaxH/{}/flt".format(name))
    except FileExistsError:
        # directory already exists
        pass
    hdu_cutout.writeto("FornaxH/{}/flt/{}.fits".format(name, name), overwrite=True)

    fieldcutout = Cutout2D(full_data, position, (cutout_size*2, cutout_size*2), wcs=wcs, copy=True)
    new_header_field = full_header.copy()
    new_header_field.update(fieldcutout.wcs.to_header())

    hdu_fieldcutout = fits.PrimaryHDU(data=fieldcutout.data, header=new_header_field)
    hdu_fieldcutout.data[cutout_size//2:cutout_size*3//2, cutout_size//2:cutout_size*3//2] = np.nan ## FIXXXX
    hdu_fieldcutout.writeto("FornaxH/{}/field_{}.fits".format(name, name), overwrite=True)
    

    plt.figure()
    plt.imshow(cutout.data, origin='lower', cmap='gray', norm='log')
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.title('FITS Cutout {}'.format(name))
    plt.show()


