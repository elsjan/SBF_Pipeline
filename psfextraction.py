import os

def extractPSF(field_path, return_path, filter):
    # the field path should be relative to where the psf-sex-files are stored
    if filter == "VIS":
        prefix = "cd psfex-sex-files-VIS-band"

    if filter == "H":
        prefix = "cd psfex-sex-files-H-band"

    command = f"{prefix} && sex ../{field_path} -CATALOG_NAME ../{return_path}/stars.cat -CATALOG_TYPE FITS_LDAC -CHECKIMAGE_NAME ../{return_path}/fieldcheck.fits"
    os.system(command)
    command = f"{prefix} && psfex ../{return_path}/stars.cat -c default.psfex"
    os.system(command)