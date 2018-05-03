import numpy
import transition_model
import matplotlib.pyplot as plt
import sys


model_params={}

model_params['lipschitz_constant']=0.3
model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.001
model_params['observation_size']=2
model_params['num_models']=4
model_params['num_epochs']=5
model_params['num_samples']=49*5
gaussian_variance=0.05
run_ID=3

fname='log/model-'+str(run_ID)+"-"+str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
	  "-"+str(gaussian_variance)+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])
tm=transition_model.neural_transition_model(model_params,True,fname)

for s in [[0,0],[3,0],[4,4],[6,2]]:
	print(s)
	li=tm.predict(numpy.array(s).reshape(1,2))
	probs=tm.probs
	for l,p in zip(li,probs):
		print(l,p)
	print("***")