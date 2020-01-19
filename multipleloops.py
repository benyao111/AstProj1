#Chi square and multiple loops
#Steps for a grid search:
#1) read in every model file in a big loop and record it’s parameter values as variables
#2) Loop over each parameter (in some range that is reasonable) (dont make too large or itll take too long to test) 
#-Read in the actual model
#-calculate and save chi squared

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import table
import pickle 
import glob



filenames = glob.glob('*.fits')
filenames = filenames[0:10]
print(filenames)


#hdu = fits.PrimaryHDU
#hdu.header


pickle_out = open("Teff.pickle","wb")
for a in range (0,10):
	hdu_list = fits.open(filenames[a])
	#data = hdu_list.info()
	hdr1 = hdu_list[0].header['TEff']
	print(hdr1)
	pickle.dump([hdr1], pickle_out)
for b in range (0,10):
	hdu_list = fits.open(filenames[b])
	#data = hdu_list.info()
	hdr2 = hdu_list[0].header['logG']
	print(hdr2)
	pickle.dump([hdr2], pickle_out)
for c in range (0,10):
	hdu_list = fits.open(filenames[c])
	#data = hdu_list.info()
	hdr3 = hdu_list[0].header['BField']
	print(hdr3)
	pickle.dump([hdr3], pickle_out)
for d in range (0,10):
	hdu_list = fits.open(filenames[d])
	#data = hdu_list.info()
	hdr4 = hdu_list[0].header['VSINI']
	print(hdr4)
	pickle.dump([hdr4], pickle_out)





#pickle.dump([hdr1,hdr2,hdr3,hdr4], pickle_out)

#pickle_out.close()
