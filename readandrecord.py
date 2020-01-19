from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import table
import pickle 
import glob,pdb
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

pickle_out = open("Teff.pickle","wb")

filenames = glob.glob('*.fits')

hdr1 = np.zeros(len(filenames))
hdr2 = np.zeros(len(filenames))
hdr3 = np.zeros(len(filenames))
hdr4 = np.zeros(len(filenames))

for f in range (len(filenames)):
	hdu_list = fits.open(filenames[f])
	#data = hdu_list.info()
	#print(hdu_list[0].header)
	hdr1[f] = hdu_list[0].header['TEff']
	hdr2[f] = hdu_list[0].header['logG']
	hdr3[f] = hdu_list[0].header['BField']
	hdr4[f] = hdu_list[0].header['VSINI']
	#print(hdr1)
	#print(hdr2)
	#print(hdr3)
	#print(hdr4)

pickle.dump((filenames,hdr1,hdr2,hdr3,hdr4), pickle_out)

pickle_out.close()

filenames,hdr1,hdr2,hdr3,hdr4 = pickle.load(open('Teff.pickle', 'rb'))
	
#the best model for TW Hya is: Teff = 3800K , log g = 4.2 , B = 3.0 kG VSINI = 5.8

data_file = np.loadtxt('mergedK.txt')
Wavelength = data_file[:,0]
NormFlux = data_file[:,1]
errors = data_file[:,2]

#need to get between 2.203097913218949 and 2.226897629358329 for wavelength

##limting to less the model wavleength range
keep = np.where((Wavelength<2.22699) & (Wavelength > 2.2030979))[0]


ReducedWavelength = data_file[keep,0]
ReducedNormFlux = data_file[keep,1]
ReducedErrors = data_file[keep,2]

filenames=filenames[0:5]

#print(ReducedWavelength)

#print(2.2/7.2e-6)

##for each file, you want to fit the best RV number:
##get mpyfit , follow instructions to install!!!!!!!!!!!!!!!

##Mpyfit need you to make a function that caluclated (model-data)/errors. This function has to look like a specific form:
##def function_name(p,args,givememodel=False):
##	data,error,wavelenth = args
##THen make some starting parameter choice startp=[70.0,p2,p3,p4]
##run mpfit
##thisfit,thisresult = mpyfit.fit(function_name,startp,(data,wavelength,error))
##thisfit.fitpars 
#thisfit.fnorm
#thisfit.perror



for f in range(len(filenames)):
	hdu_list = fits.open(filenames[f])
	mwav,mflx = hdu_list[0].data[0],hdu_list[0].data[1]
	#print(len(hdu_list))
	plt.xlabel('Wavelength (microns)')
	plt.ylabel('Norm. Flux')
	plt.plot(ReducedWavelength,ReducedNormFlux, 'b', linewidth=0.8)
	##We now know the model goes from 2.203-2.226um, 

	plt.plot(mwav/10000,mflx/np.median(mflx)*np.median(ReducedNormFlux),'g')
	nflux1, wlprime1 = pyasl.dopplerShift(mwav/10000, mflx/np.median(mflx)*np.median(ReducedNormFlux), 70., edgeHandling="fillValue", fillValue=np.median(NormFlux))
	#nflux1, wlprime1 = pyasl.dopplerShift(ReducedWavelength, ReducedNormFlux, 30., edgeHandling="fillValue", fillValue=np.median(NormFlux))
	#nflux2, wlprime = pyasl.dopplerShift(ReducedWavelength, nflux1, -30., \
                        #edgeHandling="fillValue", fillValue=np.median(NormFlux))
	nflux2, wlprime = pyasl.dopplerShift(mwav, nflux1, -60., \
                        edgeHandling="fillValue", fillValue=np.median(NormFlux))
	indi = np.arange(len(mflx)-200) + 100
	#indi = np.arange(len(ReducedNormFlux)-200) + 100
	print("Maximal difference (without outer 100 bins): ", \
                #max(np.abs(ReducedNormFlux[indi]-nflux2[indi])))
                max(np.abs(mflx[indi]-nflux2[indi])))

#x = np.arange(len(dwav))
#interp = np.interp(x, Wavelength, Wavelength)
##interpolate the model onto the wavelength points of the data
	interpmodel = np.interp(ReducedWavelength, mwav, mflx)
	#print (interpmodel)
	#print (len(interpmodel))
#print (Wavelength - dwav)
#chi2 = chisquare(interp, dwav)
	residuals = (interpmodel-ReducedNormFlux)/ReducedErrors
	chi2 = np.sum(residuals**2) 
	print (chi2)
	#plt.figure()
	plt.title("Initial (blue), shifted (red), and back-shifted (green) spectrum")
	#plt.plot(ReducedWavelength, ReducedNormFlux/np.median(ReducedNormFlux))
	#plt.plot(ReducedWavelength, (ReducedNormFlux + nflux1)/np.median(ReducedNormFlux+nflux1))
	#plt.plot(mwav/10000, mflx/np.median(mflx),'g')
	plt.plot(mwav/10000, (nflux1)/np.median(nflux1)*np.median(ReducedNormFlux), 'r')
	plt.xlim((2.20, 2.230))
	plt.show()

#THIS WEEK: 7/1/19
#Model = R(MxA+B) load in
#
#scale (A) and continuum (B)
#R= delta RV(w)
#RV shift model to match data (use code from last time)
#after that play with A and B to try and eyebal and get it to match/ i guess line up?





#Steps for a grid search:
#1) read in every model file in a big loop and record it’s parameter values as variables
#2) Loop over each parameter (in some range that is reasonable) (dont make too large or itll take too long to test) 
#-Read in the actual model
#-calculate and save chi squared

#Saving variables in python
#Import pickle
#To save some variables:
#pickle.dump((var1,var2….),open(‘filename’,’w’))
#To read back in 
#var1,var2… = pickle.load(open(‘filename’,’r’))
