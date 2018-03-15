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


def f0(x):
	return numpy.tanh(x)+3
def f1(x):
	return x*x
def f2(x):
	return numpy.sin(x)-5
def f3(x):
	return numpy.sin(x)-3
def f4(x):
	return numpy.sin(x)*numpy.sin(x)


def create_train_data(li_num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(li_num_samples[l]):
			sample=numpy.random.uniform(-2,2)
			if l==0:
				li_samples.append(sample)
				label=f0(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_labels.append(label)
					
			elif l==1:
				label=f1(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==2:
				label=f2(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==3:
				label=f3(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==4:
				label=f4(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			else:
				print('not implemented yet ... aborting')
				sys.exit(1)
			#print(sample,label)
	return li_samples,li_labels

def wasserstein(true_probs,true_labels,estimated_probs,estimated_labels):
	W=scipy.stats.wasserstein_distance(true_labels, estimated_labels, u_weights=true_probs, v_weights=estimated_probs)
	return W

def compute_wass_loss(tm,phi,em_object):
	sample_li=numpy.random.uniform(-2,2,100)
	total_wass=0
	for x in sample_li:
		true_labels=[f0(x),f1(x),f2(x),f3(x),f4(x)]
		true_probs=5*[0.2]
		estimated_labels=[m.predict(numpy.array(x).reshape(1,1))[0,0] for m in tm.models]
		estimated_probs=em_object.learned_priors
		total_wass=total_wass+wasserstein(true_probs,true_labels,estimated_probs,estimated_labels)
	#print("Wasserstein loss",total_wass)
	return total_wass
	#sys.exit(1)


def create_matrices(li_samples,li_labels):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,1)
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,1)
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels


def plot_everything(li_samples,li_labels,tm,phi,ax):
	ax.clear()
	ax.plot(li_samples,li_labels,'o')
	y_li=tm.predict(phi)
	for index,y in enumerate(y_li):
		ax.plot(li_samples,y,'o',lw=2,label=index)
	plt.legend()






num_experiments=1
li_num_samples=5*[30]
num_lines=len(li_num_samples)
plot=False

model_params={}
try:
	model_params['lipschitz_constant']=float(sys.argv[2])
except:
	model_params['lipschitz_constant']=.2
model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.0001
model_params['observation_size']=1
model_params['num_models']=num_lines
model_params['num_epochs']=50
em_params={}
em_params['num_iterations']=100
try:
	em_params['gaussian_variance']=float(sys.argv[3])
except:
	em_params['gaussian_variance']=.05
em_params['num_models']=model_params['num_models']
if plot==True:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
li_obj_all=[]
li_w=[]
li_em_obj=[]
li_samples,li_labels=create_train_data(li_num_samples,num_lines)
phi,y=create_matrices(li_samples,li_labels)
tm=transition_model.neural_transition_model(model_params)
em_object=em.em_learner(em_params)
for iteration in range(em_params['num_iterations']):
	li_em_obj.append(em_object.e_step_m_step(tm,phi,y,iteration))
	if plot==True:
		plot_everything(li_samples,li_labels,tm,phi,ax)
		plt.pause(.5)
	li_w.append(compute_wass_loss(tm,phi,em_object))
	print("li_w",li_w)
	print("li_em_obj",li_em_obj)
	sys.stdout.flush()
	numpy.savetxt("w_loss-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_w)
	numpy.savetxt("em_obj-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_w)

