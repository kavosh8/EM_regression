import numpy
import matplotlib.pyplot
for variance in [0.1,0.05,0.01]:
	for k in [0.1,0.15,0.2,0.25,0.35,0.5,1.0,2.0]:
		temp_li=[]
		for run_ID in range(25):
			temp=numpy.loadtxt('log/w_loss-'+str(run_ID)+"-"+
				  str(k)+"-"+
			 	  str(variance)+".txt")
			temp_li.append(temp)
		print(k,variance,numpy.mean(temp_li[-1],axis=0))
