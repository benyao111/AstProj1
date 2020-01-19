from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt
import pdb
hdu_list = fits.open('conv_T3200_G4.0_B0.0_M0.00_t2.0_R45000_V0.0.fits')
hdu_list.info()
image_data = hdu_list[0].data 

wav,flx=image_data[0,:],image_data[1,:]
print(wav.shape,flx.shape)


#print(hdu_list[0].header)
#modteff=hdu_list[0].header['TEFF']
#print(modteff)


