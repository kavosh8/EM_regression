import numpy
import matplotlib.pyplot as plt
stds=[]
means=[]
for types in ['tabular','deterministic','stochastic']:
	temp=numpy.loadtxt(types+".txt")
	means.append(numpy.mean(temp))
	stds.append(numpy.std(temp))
plt.errorbar(x=[1,2,3],y=means,yerr=stds)
plt.xticks(numpy.arange(1,4), ('tabular','deterministic','stochastic'))
plt.ylabel("number of deaths in 1000 steps")
plt.show()
plt.close()