import numpy

for num_samples in [1000,2000]:
	for learning_rate in [0.005,0.001,0.0005]:
		for gaussian_variance in [0.1,0.05]:
			for num_hidden_layers in [0,1,2]:
				li_runs=[]
				for run_number in range(10):
					try:
						fname="log/w_loss-"+str(run_number)+"-"+str(num_samples)+"-"+str(learning_rate)+"-"+str(gaussian_variance)+"-"+str(run_number)+".txt"
						temp=numpy.loadtxt(fname)
						if len(temp)==500:
							li_runs.append(temp)
					except:
						pass
				print("hyperparameters: ",num_samples,learning_rate,gaussian_variance,run_number)
				print("average and min over {} runs is {}".format(len(li_runs),numpy.mean(li_runs,axis=0)[-1]))