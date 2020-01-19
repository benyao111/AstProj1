import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

#data_file = np.loadtxt('mergedH.txt')
#Wavelength = data_file[:,0]
#NormFlux = data_file[:,1]

data_file2 = np.loadtxt('mergedk.txt')
Wavelength2 = data_file2[:,0]
NormFlux2 = data_file2[:,1]

plt.xlabel('Wavelength (microns)')
plt.ylabel('Norm. Flux')
#plt.plot(Wavelength, NormFlux, 'r')

plt.plot(Wavelength2, NormFlux2, 'b')
plt.xlim((2.15, 2.35))
plt.ylim((0, 2))
plt.show()