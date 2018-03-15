import numpy
import matplotlib.pyplot
import matplotlib.pyplot as plt
li_k=[0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.5,0.75,1.0]
li_w={}
li_em_obj={}
for variance in [0.05]:
	for k in li_k:
		temp_li=[]
		for run_ID in range(10):
			try:
				temp=numpy.loadtxt('log/w_loss-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				if len(temp)==100:
					temp_li.append(temp)
			except:
				pass
		li_w[k]=temp_li

		temp2_li=[]
		for run_ID in range(10):
			try:
				temp=numpy.loadtxt('log/em_obj-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				if len(temp)==100:
					temp2_li.append(temp)
			except:
				pass
		li_em_obj[k]=temp2_li

plt.subplot(221)
for k in li_k:
	plt.plot(numpy.mean(li_w[k],axis=0))

plt.subplot(222)
for k in li_k:
	plt.plot(numpy.mean(li_em_obj[k],axis=0))

plt.show()
#plt.legend()
#plt.show()
#print(li_w)
#plt.close()
