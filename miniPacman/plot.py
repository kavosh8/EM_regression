import numpy

for num_samples in [2000]:
	for learning_rate in [0.001,0.0005]:
		for gaussian_variance in [0.05]:
			for num_hidden_layers in [0,1,2]:
				for lipschitz_constant in [0.1,0.2,0.25,0.5,1.,1.5,2.]:
					li_runs=[]
					for run_number in range(5):
						try:
							fname="log/w_loss-"+str(run_number)+"-"+str(num_samples)+"-"+\
							str(learning_rate)+"-"+str(gaussian_variance)+"-"+str(num_hidden_layers)+"-"+str(lipschitz_constant)+".txt"
							temp=numpy.loadtxt(fname)
							if len(temp)==500:
								li_runs.append(temp)
						except:
							pass
					print("hyperparameters: ",num_samples,learning_rate,gaussian_variance,num_hidden_layers,lipschitz_constant)
					print("average over {} runs is {}".format(len(li_runs),numpy.mean(li_runs,axis=0)[-1]))