import numpy
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import transition_model
import math

def create_train_data(num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(num_samples):
			sample=numpy.random.uniform(-1,1)
			if l==0:
				label=numpy.sin(3*sample)+1+numpy.random.normal(loc=0.0, scale=0.0,size=1)
			elif l==1:
				label=numpy.cos(3*sample)+numpy.random.normal(loc=0.0, scale=0.0,size=1)
			elif l==2:
				label=numpy.sin(3*sample)*numpy.cos(3*sample)+numpy.random.normal(loc=0.0, scale=0.0,size=1)
			else:
				print('not implemented yet ... aborting')
				sys.exit(1)
			li_samples.append(sample)
			li_labels.append(label[0])
			#print(sample,label)
	return li_samples,li_labels
def create_matrices(li_samples,li_labels):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,1)
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,1)
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels

def compute_posterior(tm,phi,y):

	o_li=[x.predict(phi) for x in tm.models]
	p_li=[]
	for o in o_li:
		p=numpy.array(y-o)
		p=numpy.exp(-numpy.multiply(p,p)/gaussian_variance).flatten()
		for x in p:
			if math.isnan(x):
				print("underflow")
				sys.exit(1)
		p_li.append(p)
	p_arr=numpy.transpose(numpy.array(p_li))
	for i in range(p_arr.shape[0]):
		sum_probs=numpy.sum(p_arr[i,:])
		for j in range(p_arr.shape[1]):
			p_arr[i,j]=p_arr[i,j]/sum_probs
	p_li=[]
	p_arr=numpy.transpose(p_arr)
	for i in range(len(p_arr)):
		p_li.append(p_arr[i,:])
	return p_li


def plot_everything(li_samples,li_labels,tm,phi,ax):
	ax.clear()
	ax.plot(li_samples,li_labels,'o')
	y_li=tm.predict(phi)
	for y in y_li:
		ax.plot(li_samples,y,'o',lw=2)


num_experiments=1
num_samples=100
num_lines=3
num_iterations=100
gaussian_variance=.01
plot=True

model_params={}
model_params['lipschitz_constant']=0.8
model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=16
model_params['activation_fn']='relu'
model_params['learning_rate']=0.00025
model_params['observation_size']=1
model_params['num_models']=3
model_params['num_epochs']=100


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
li_obj_all=[]
for experiment in range(num_experiments):
	li_samples,li_labels=create_train_data(num_samples,num_lines)
	phi,y=create_matrices(li_samples,li_labels)
	tm=transition_model.neural_transition_model(model_params)
	for iteration in range(num_iterations):
		w_li=compute_posterior(tm,phi,y)#E step
		if iteration>0:
			tm.regression(phi,y,w_li)#M step
		if plot==True:
			plot_everything(li_samples,li_labels,tm,phi,ax)
			plt.pause(.1)

