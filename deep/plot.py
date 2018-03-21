import numpy
import matplotlib.pyplot
import matplotlib.pyplot as plt
li_k=[0.05,0.075,0.1,0.125,0.15,0.175,0.25,0.3,0.35,0.5,0.75,1.0,2.0]
li_w={}
li_em_obj={}
num_runs_used=200
for variance in [0.05]:
	for k in li_k:
		temp_li=[]
		for run_ID in range(num_runs_used):
			try:
				temp=numpy.loadtxt('log/w_loss-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				temp_li.append(temp)
			except:
				pass
		li_w[k]=temp_li

		temp2_li=[]
		for run_ID in range(num_runs_used):
			try:
				temp=numpy.loadtxt('log/em_obj-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				temp2_li.append(temp)
			except:
				pass
		li_em_obj[k]=temp2_li

plt.subplot(222)
for k in li_k:
	plt.plot(numpy.mean(li_w[k],axis=0),label="k="+str(k))
plt.xlabel('iterations')
plt.ylabel('Wass')
plt.ylim([40,200])
plt.legend(fontsize=8)

plt.subplot(221)
for k in li_k:
	plt.plot(numpy.mean(li_em_obj[k],axis=0),label="k="+str(k))
plt.xlabel('iterations')
plt.ylabel('EM loss')
plt.ylim([-10000,0])
plt.legend(fontsize=8)

plt.subplot(224)
#print([li_w[0.175][x][-1] for x in range(200)])
y=[numpy.mean(li_w[x],axis=0)[-1] for x in li_k]
plt.plot(li_k,y,lw=4)
plt.xlabel('k')
plt.ylabel('Wass final')

plt.subplot(223)
y=[numpy.mean(li_em_obj[x],axis=0)[-1] for x in li_k]
plt.plot(li_k,y,lw=4)
plt.xlabel('k')
plt.ylabel('EM final')
plt.show()
#plt.legend()
#plt.show()
#print(li_w)
#plt.close()
