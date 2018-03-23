import numpy
import numpy.random
import sys
run_ID=int(sys.argv[1])
numpy.random.seed(run_ID)
import tensorflow as tf
tf.set_random_seed(run_ID)
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import transition_model
import math
import em
import keras
import scipy.stats
import time
import csv
import utils


model_params={}
model_params['lipschitz_constant']=1.1
model_params['num_hidden_layers']=0
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.0005
model_params['observation_size']=4
model_params['num_models']=16
model_params['num_epochs']=5
model_params['num_samples']=100*16

em_params={}
em_params['num_iterations']=500
em_params['gaussian_variance']=.01#for 1D problem, effective range is 0.25 to 0.001
em_params['num_models']=model_params['num_models']
em_params['observation_size']=model_params['observation_size']


li_w,li_em_obj=[],[]
#build training data
li_samples,li_labels=utils.load_synthetic_data(model_params['num_samples'])
phi,y=utils.create_matrices(li_samples,li_labels,model_params)
#create transition model
tm=transition_model.neural_transition_model(model_params)
#create em object
em_object=em.em_learner(em_params)

for iteration in range(em_params['num_iterations']):
	li_em_obj.append(em_object.e_step_m_step(tm,phi,y,iteration))# do one EM iteration
	li_w.append(utils.compute_approx_wass_loss(tm,em_object))
	print(iteration,"li_em_obj:",li_em_obj[-1],"li_w",li_w[-1])#print EM objective
	sys.stdout.flush()
	if iteration%20==0 and iteration>0:
		for number,x in enumerate(tm.predict(numpy.array([3,3,3,3]).reshape(1,model_params['observation_size']))):
			print("number:",number,"next state:",x.tolist(),"prob:",em_object.learned_priors[number])#print Wasserstein objective

