import numpy as np
import pdb

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

##what yours might look like:

#def logprob_rv(p,wavelength,flux,modwav,modflux,givememodel=False):
	

	##do all the model things
	##shift the rv
	##calcuate resuds
	##if givememodel == True:
#		return final model
#	else:
#		return logprob


def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5*np.dot(diff, np.linalg.solve(cov,diff))

ndim = 5

np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)


nwalkers = 10
p0 = np.random.rand(nwalkers, ndim)*100
import emcee

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

print(log_prob(p0[0], means, cov))

state = sampler.run_mcmc(p0, 1)
sampler.reset()

sampler.run_mcmc(state, 10000,progress=True)

pdb.set_trace()

import matplotlib.pyplot as plt



samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])

plt.show()

pdb.set_trace()