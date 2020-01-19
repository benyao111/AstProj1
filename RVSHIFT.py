#New Data Plot

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

data_file2 = np.loadtxt('mergedk.txt')
Wavelength = data_file2[:,0]
NormFlux = data_file2[:,1]
error = data_file2[:,2]

plt.xlabel('Wavelength (microns)')
plt.ylabel('Norm. Flux')

#wave_log = np.min(Wavelength2)*np.exp(np.log(np.max(Wavelength2)/np.min(Wavelength2))/40*np.arange(wave_log))
#drv = np.log(wave_log[1]/wave_log[0])*2.998e5
#plt.plot (wave_log, NormFlux2, 'b')


# Create a "spectrum" with 0.01 A binning ...
#wvl = np.linspace(6000., 6100., 10000)
# ... a gradient in the continuum ...
#flux = np.ones(len(Wavelength)) + (Wavelength/Wavelength.min())*0.05
# ... and a Gaussian absoption line
#flux -= np.exp( -(Wavelength-6050.)**2/(2.*0.5**2) )*0.05

# Shift that spectrum redward by 30 km/s using
# "firstlast" as edge handling method.
nflux1, wlprime1 = pyasl.dopplerShift(Wavelength, NormFlux, 30., edgeHandling="fillValue", fillValue=np.median(NormFlux))

# Shift the red-shifted spectrum blueward by 30 km/s, i.e.,
# back on the initial spectrum.
nflux2, wlprime = pyasl.dopplerShift(Wavelength, nflux1, -30., \
                        edgeHandling="fillValue", fillValue=np.median(NormFlux))

# Check the maximum difference in the central part
indi = np.arange(len(NormFlux)-200) + 100
print("Maximal difference (without outer 100 bins): ", \
                max(np.abs(NormFlux[indi]-nflux2[indi])))

# Plot the outcome
plt.title("Initial (blue), shifted (red), and back-shifted (green) spectrum")
plt.plot(Wavelength, NormFlux/np.median(NormFlux))
#plt.plot(Wavelength, nflux1, 'r.-')
plt.plot(Wavelength, (NormFlux + nflux1)/np.median(NormFlux+nflux1))
#plt.plot(Wavelength, nflux2, 'g.-')
plt.xlim((2.20, 2.30))
#plt.ylim((0, 2))
plt.show()