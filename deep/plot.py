import numpy
import matplotlib.pyplot
import matplotlib.pyplot as plt
li_k=[0.05,0.1,0.15,0.2,0.25,0.35,0.5,0.75,1.0,2.0]
li_w=[]
for variance in [0.05]:
	for k in li_k:
		temp_li=[]
		for run_ID in range(200):
			try:
				temp=numpy.loadtxt('log/w_loss-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				if len(temp)==100:
					temp_li.append(temp)
			except:
				#print(variance,k,run_ID,'not found')
				pass
		print(k,variance,len(temp_li),numpy.mean(temp_li,axis=0)[-1])
		li_w.append(numpy.mean(temp_li,axis=0)[-1])
		plt.plot(numpy.mean(temp_li,axis=0),label=k)
plt.legend()
plt.show()
print(li_w)
plt.close()
