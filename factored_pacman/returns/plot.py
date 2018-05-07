import numpy
import matplotlib.pyplot as plt

x=[0.2,0.25,0.3,0.5,1.0]
y=[]
y_std=[]
ax = plt.subplot(111)
for lipschitz_constant in x:
	li=[]
	for run_number in range(5):
		for learning_rate in [0.005,0.001]:
			for gaussian_variance in [0.1,0.05]:
				temp=numpy.loadtxt('stochastic/'+\
					str(lipschitz_constant)+"-"+str(run_number)+"-"+str(learning_rate)+"-"+str(gaussian_variance)+'.txt')
				li.append(numpy.mean(temp))
	y.append(-numpy.mean(li))
	y_std.append(numpy.std(li)/10)
ax.errorbar(x,y,yerr=y_std,lw=3,color='red')
tabular_result=numpy.loadtxt('other/tabular.txt')
ax.plot(x,5*[-numpy.mean(tabular_result)],'--',label='tabular baseline')
print(numpy.mean(tabular_result))
'''
random_result=numpy.loadtxt('other/random.txt')[0]
plt.plot(x,5*[random_result])
'''
det_result=numpy.loadtxt('other/deterministic.txt')
print(numpy.mean(det_result))
ax.plot(x,5*[-numpy.mean(det_result)],'--',label='deterministic baseline')
plt.xticks([0.2,0.5,0.7,1])
plt.yticks([-14,-11,-9])
plt.xlim([0.15,1.1])
plt.xlabel("Lipschitz constant",size=14)
plt.ylabel("return",rotation=0,size=14)


# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.legend()
#plt.yscale('log')
plt.show()
plt.close()
