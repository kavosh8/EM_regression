import numpy
import numpy.random
import sys
run_number=int(sys.argv[1])
numpy.random.seed(run_number)
import tensorflow as tf
tf.set_random_seed(run_number)
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



# take as input, from qsub script, run ID num samples stepsize and gaussian variance ...

model_params={}
try:
	model_params['lipschitz_constant']=float(sys.argv[6])
except:
	model_params['lipschitz_constant']=.3
try:
	model_params['num_hidden_layers']=int(sys.argv[5])
except:
	print("num hidden_layer_nodes not found setting it to 0")
	model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
try:
	model_params['learning_rate']=float(sys.argv[3])
except:
	print("learning rate not found setting it to 0.005")
	model_params['learning_rate']=0.005
model_params['observation_size']=2
model_params['num_models']=4
model_params['num_epochs']=5
try:
	model_params['num_samples']=int(sys.argv[2])
except:
	print("num samples not found .. setting it to 1000")
	model_params['num_samples']=5*49

em_params={}
em_params['num_iterations']=500
try:
	em_params['gaussian_variance']=float(sys.argv[4])#for 1D problem, effective range is 0.25 to 0.001
except:
	print("gaussian variance not found .. setting it to 0.1")
	em_params['gaussian_variance']=.1

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
	if iteration%10==0 and iteration>0:
		print([3,3])
		for number,x in enumerate(tm.predict(numpy.array([3,3]).reshape(1,model_params['observation_size']))):
			print("number:",number,"next state:",x.tolist(),"prob:",em_object.learned_priors[number])#print Wasserstein objective
		print([[0,0]])
		for number,x in enumerate(tm.predict(numpy.array([0,0]).reshape(1,model_params['observation_size']))):
			print("number:",number,"next state:",x.tolist(),"prob:",em_object.learned_priors[number])#print Wasserstein objective
		print([[2,0]])
		for number,x in enumerate(tm.predict(numpy.array([2,0]).reshape(1,model_params['observation_size']))):
			print("number:",number,"next state:",x.tolist(),"prob:",em_object.learned_priors[number])#print Wasserstein objective
		print([6,0])
		for number,x in enumerate(tm.predict(numpy.array([6,0]).reshape(1,model_params['observation_size']))):
			print("number:",number,"next state:",x.tolist(),"prob:",em_object.learned_priors[number])#print Wasserstein objective
	numpy.savetxt("log/w_loss-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])+".txt",li_w)
	numpy.savetxt("log/em_loss-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])+".txt",li_em_obj)
	li_show=[]
	for index,m in enumerate(tm.models):
		mname="log/model-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+\
				 "-"+str(model_params['lipschitz_constant'])+"-"+str(index)+".h5"
		m.save_weights(mname)



