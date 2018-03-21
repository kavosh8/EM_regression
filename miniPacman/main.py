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

def load_data(fname):
	with open(fname, 'rb') as f:
		data = list(csv.reader(f))
	out=[]
	for d in data:
		temp=[]
		for s in d:
			temp.append(float(s))
		out.append(temp)
	return out

def load_synthetic_data(N):
	li_s,li_sprime=[],[]
	for _ in range(N):
		x1,y1,x2,y2=numpy.random.randint(1,10,4)
		s=[x1,y1,x2,y2]
		case=numpy.random.randint(0,8)
		if case==0:
			s_p=[x1+1,y1,x2+1,y2]
		elif case==1:
			s_p=[x1+1,y1,x2-1,y2]
		elif case==2:
			s_p=[x1+1,y1,x2,y2+1]
		elif case==3:
			s_p=[x1+1,y1,x2,y2-1]
		elif case==4:
			s_p=[x1,y1+1,x2+1,y2]
		elif case==5:
			s_p=[x1,y1+1,x2-1,y2]
		elif case==6:
			s_p=[x1,y1-1,x2+1,y2]
		elif case==7:
			s_p=[x1,y1-1,x2-1,y2]
		li_s.append(s)
		li_sprime.append(s_p)
	#print(li_s)
	#print(li_sprime)
	#sys.exit(1)
	return li_s,li_sprime

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


def create_matrices(li_samples,li_labels,model_params):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,model_params['observation_size'])
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,model_params['observation_size'])
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
num_lines=8
plot=False

model_params={}
try:
	model_params['lipschitz_constant']=float(sys.argv[2])
except:
	model_params['lipschitz_constant']=.2
model_params['num_hidden_layers']=1
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.0001
model_params['observation_size']=4
model_params['num_models']=num_lines
model_params['num_epochs']=10
em_params={}
em_params['num_iterations']=100
try:
	em_params['gaussian_variance']=float(sys.argv[3])
except:
	em_params['gaussian_variance']=.01
em_params['num_models']=model_params['num_models']
em_params['observation_size']=model_params['observation_size']
if plot==True:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
li_obj_all=[]
li_w=[]
li_em_obj=[]
#li_samples,li_labels=load_data('states.csv'),load_data('next_states.csv')
li_samples,li_labels=load_synthetic_data(5000)

phi,y=create_matrices(li_samples,li_labels,model_params)
tm=transition_model.neural_transition_model(model_params)
em_object=em.em_learner(em_params)

for iteration in range(em_params['num_iterations']):
	li_em_obj.append(em_object.e_step_m_step(tm,phi,y,iteration))
	for number,x in enumerate(tm.predict(numpy.array([2,3,5,3]).reshape(1,4))):
		print(number,x)
	#sys.exit(1)
	'''
	if plot==True:
		plot_everything(li_samples,li_labels,tm,phi,ax)
		fig.savefig('save/visualize'+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+'iteration-'+str(iteration)+'.pdf')
		if iteration==0:
			plt.pause(25)
		else:
			plt.pause(.5)
	'''
	#li_w.append(compute_wass_loss(tm,phi,em_object))
	#print("li_w",li_w)
	print("li_em_obj",li_em_obj)
	sys.stdout.flush()
	'''
	numpy.savetxt("w_loss-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_w)
	numpy.savetxt("em_obj-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_em_obj)
	'''

