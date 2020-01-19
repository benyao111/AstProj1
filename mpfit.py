from __future__ import print_function
import numpy as np
import pdb
import emcee
import matplotlib.pyplot as plt
import pandas as pd 
from astropy.io import fits
from astropy.table import table
import pickle 
import glob,pdb
from scipy.stats import chisquare
from PyAstronomy import pyasl

####################################

pickle_out = open("Teff.pickle","wb")

filenames = glob.glob('*.fits')

hdr1 = np.zeros(len(filenames))
hdr2 = np.zeros(len(filenames))
hdr3 = np.zeros(len(filenames))
hdr4 = np.zeros(len(filenames))

for f in range (len(filenames)):
	hdu_list = fits.open(filenames[f])
	hdr1[f] = hdu_list[0].header['TEff']
	hdr2[f] = hdu_list[0].header['logG']
	hdr3[f] = hdu_list[0].header['BField']
	hdr4[f] = hdu_list[0].header['VSINI']


pickle.dump((filenames,hdr1,hdr2,hdr3,hdr4), pickle_out)

pickle_out.close()

filenames,hdr1,hdr2,hdr3,hdr4 = pickle.load(open('Teff.pickle', 'rb'))

data_file = np.loadtxt('mergedK.txt')
Wavelength = data_file[:,0]
NormFlux = data_file[:,1]
errors = data_file[:,2]

keep = np.where((Wavelength<2.22699) & (Wavelength > 2.2030979))[0]

ReducedWavelength = data_file[keep,0]
ReducedNormFlux = data_file[keep,1]
ReducedErrors = data_file[keep,2]

filenames=filenames[0:5]

############################################

for f in range(len(filenames)):
	hdu_list = fits.open(filenames[f])
	mwav,mflx = hdu_list[0].data[0],hdu_list[0].data[1]
	plt.xlabel('Wavelength (microns)')
	plt.ylabel('Norm. Flux')
	plt.plot(ReducedWavelength,ReducedNormFlux, 'b', linewidth=0.8)
	plt.plot(mwav/10000,mflx/np.median(mflx)*np.median(ReducedNormFlux),'g')
	nflux1, wlprime1 = pyasl.dopplerShift(mwav/10000, mflx/np.median(mflx)*np.median(ReducedNormFlux), 70., edgeHandling="fillValue", fillValue=np.median(NormFlux))
	nflux2, wlprime = pyasl.dopplerShift(mwav, nflux1, -60., \
                        edgeHandling="fillValue", fillValue=np.median(NormFlux))
	indi = np.arange(len(mflx)-200) + 100
	print("Maximal difference (without outer 100 bins): ", \
                max(np.abs(mflx[indi]-nflux2[indi])))
	interpmodel = np.interp(ReducedWavelength, mwav, mflx)
	residuals = (interpmodel-ReducedNormFlux)/ReducedErrors
	chi2 = np.sum(residuals**2) 
	#print (chi2)
	#plt.title("Initial (blue), shifted (red), and back-shifted (green) spectrum")
	#plt.plot(mwav/10000, (nflux1)/np.median(nflux1)*np.median(ReducedNormFlux), 'r')
	#plt.xlim((2.20, 2.230))
	#plt.show()

#######################################

def logprob_rv(p, ReducedWavelength, ReducedNormFlux, ReducedErrors, mwav, mflux, givememodel=False):
	logprob = np.sum(-(ReducedNormFlux-mflx**2/2/ReducedErrors**2) + 1/sqrt(2*pi*ReducedErrors**2))
	return logprob (p, Wavelength, NormFlux, ReducedErrors, mwav, mflx, givememodel=False)

ndim = 2

nwalkers = 10
p0 = np.random.rand(nwalkers, ndim)

testmodel = logprob_rv([-30.0],ReducedWavelength,ReducedNormFlux,ReducedErrors,mwav,mflx,givememodel=True)

#said it doesnt work bc i need interpmodel.

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

sampler.run_mcmc(p0, 10000,progress=True)
##pickle dump sampler.get_chain()

pdb.set_trace()



samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])

plt.show()
##Structure: we want to produce a model with parameters given in p, 
#and then computer logprob when compared to the real data
## interpolate onto the wavelenght points of the data
##thats it you now have modflux!

	##logprob = np.sum(-(flux-modflox)**2/2/error**2) + 1/sqrt(2*pi*errors^2)
	##return logprob
## make modflux: read it in from a file
## rv shift (based on one of the p parameters)
##basically we want a normal distribution graph and the middle part (apex of the graph should tell us what the ideal rv shift is)
	##do all the model things
	##shift the rv
	##calcuate residuals
	##if givememodel == True:
#		return modflux
#	else:
#		return logprob