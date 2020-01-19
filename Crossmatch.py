import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from gcirc import gcirc
import pdb
filename = 'fields.zip'
Greglist = pd.read_csv(filename.open,sep=',')
#print(Greglist.columns)
#print(Greglist['RA.1'].values)

filename2 = 'bin_rv_mon.csv'
Aaronlist = pd.read_csv(filename2)
#print(Aaronlist.columns)
#print(Aaronlist.RA.values)

##Loop over gregs list of observations and format the RA's into decimals
#print(Aaronlist.RA.values[-1].split(' '))
aaronra = np.zeros(len(Aaronlist.RA.values),dtype=float)
aarondec = np.zeros(len(Aaronlist.DEC.values), dtype=float)

outfile=open('list_for_greg.txt','w')
outfile.write('Starname Filename Date \n')
outfile.flush()
for i in range(len(Aaronlist.RA.values)):
	thisRA = Aaronlist.RA.values[i]
	thisrah = int(thisRA.split(' ')[0])
	thisram = int(thisRA.split(' ')[1])
	thisras = float(thisRA.split(' ')[2][0:5])
	aaronra[i] =thisrah*15.0 + thisram/60.0*15 + thisras/60.0/60.0*15.0
	thisDEC = Aaronlist.DEC.values[i]
	thisdecdeg = abs(int(thisDEC.split(' ')[0]))
	thisdecmin = int(thisDEC.split(' ')[1])
	thisdecsec = float(thisDEC.split(' ')[2]) 
	aarondec[i] = thisdecdeg + thisdecmin/60.0 + thisdecsec/3600.0 
	if thisDEC[0] == '-': aarondec[i]*=-1.0
	distance = gcirc(aaronra[i],aarondec[i],Greglist['RA.1'].values,Greglist['DEC.1'].values)
	toout=np.where(distance < 10)[0]
	#pdb.set_trace()
	##find all files closer than 10 arcsec

	##now do a second loop, and print each file to a line in the outfile text file.
	for j in range(len(toout)):
		#make a big line by combining the filename from gregs file, and starname from aaronlist, and date from gregs list
		#Greglist.filename.values[toout[j]]
		FILENAME = Greglist.FILENAME.values[toout[j]]
		OBJNAME = Greglist.OBJNAME.values[toout[j]]
		#OBJNAMEAARON = Aaronlist.Star.values[i]
		#JULIANDATE = str(Greglist.JD.values[toout[j]])
		CIVILDATE = str(Greglist.CIVIL.values[toout[j]])
		#thisline = (Greglist.FILENAME.values[toout[j]] Greglist.OBJNAME.values[toout[j]] str(Greglist.CIVIL.values[toout[j]]) + '\n')
		thisline = (OBJNAME + ',' + FILENAME + ',' +  CIVILDATE + '\n')
		outfile.write(thisline)

outfile.close()




	##then compare the RA/DEC to gregs numbers and find the relted observations
	#save those filenmaes somewhere, i.e. in another variable.



