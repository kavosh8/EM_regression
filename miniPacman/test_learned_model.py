import numpy
import transition_model
import matplotlib.pyplot as plt



model_params={}

model_params['lipschitz_constant']=0.25
model_params['num_hidden_layers']=1
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.001
model_params['observation_size']=4
model_params['num_models']=16
model_params['num_epochs']=5
model_params['num_samples']=3000
gaussian_variance=0.05
run_ID=4

fname='best_models/model-'+str(run_ID)+"-"+str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
	  "-"+str(gaussian_variance)+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])
tm=transition_model.neural_transition_model(model_params,True,fname)

for dimension_number in range(model_params['observation_size']):
	plt.subplot(str(22)+str(dimension_number))
	all_li=[]
	s_li=range(8)
	for _ in range(model_params['num_models']):
		all_li.append([])
	for s in s_li:
		temp=[3,3,3,3]
		temp[dimension_number]=s
		x=numpy.array(temp).reshape((1,4))
		x_prime=tm.predict(x)
		for z,xp in enumerate(x_prime):
			all_li[z].append(xp[0][dimension_number])
	for y in all_li:
		plt.plot(s_li,y)
plt.show()
plt.close()