import numpy
import matplotlib.pyplot as plt

for num_samples in [1000,2000,3000]:
	means=[]
	for learning_rate in [0.001]:
		for gaussian_variance in [0.05]:
			for num_hidden_layers in [1]:
				for lipschitz_constant in [0.1,0.2,0.25,0.3,0.5]:
					li_runs=[]
					for run_number in range(10):
						try:
							fname="log/w_loss-"+str(run_number)+"-"+str(num_samples)+"-"+\
							str(learning_rate)+"-"+str(gaussian_variance)+"-"+str(num_hidden_layers)+"-"+str(lipschitz_constant)+".txt"
							temp=numpy.loadtxt(fname)
							if len(temp)==500:
								li_runs.append(temp)
						except:
							pass
					means.append(numpy.mean(li_runs,axis=0)[-1])
				plt.plot([0.1,0.2,0.25,0.3,0.5],means,label='num training samples: '+str(num_samples))
plt.xlabel("Lipschitz constant")
plt.ylabel("Wasserstein loss")
plt.legend()
plt.show()