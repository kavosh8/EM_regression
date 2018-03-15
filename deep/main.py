import numpy
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import transition_model
import math
import em

def create_train_data(li_num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(li_num_samples[l]):
			sample=numpy.random.uniform(-2,2)
			if l==0:
				li_samples.append(sample)
				label=numpy.tanh(sample)+3#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_labels.append(label)
					
			elif l==1:
				label=sample*sample#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==2:
				label=numpy.sin(sample)-5#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==3:
				label=numpy.sin(sample)-3#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==4:
				label=numpy.sin(sample)*numpy.sin(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			else:
				print('not implemented yet ... aborting')
				sys.exit(1)
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


def plot_everything(li_samples,li_labels,tm,phi,ax):
	ax.clear()
	ax.plot(li_samples,li_labels,'o')
	y_li=tm.predict(phi)
	for index,y in enumerate(y_li):
		ax.plot(li_samples,y,'o',lw=2,label=index)
	plt.legend()


num_experiments=1
li_num_samples=[20,20,20,20,20]
num_lines=len(li_num_samples)
plot=True


model_params={}
model_params['lipschitz_constant']=.2
model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.0001
model_params['observation_size']=1
model_params['num_models']=num_lines
model_params['num_epochs']=100

em_params={}
em_params['num_iterations']=200
em_params['gaussian_variance']=.05
em_params['num_models']=model_params['num_models']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
li_obj_all=[]
for experiment in range(num_experiments):
	li_samples,li_labels=create_train_data(li_num_samples,num_lines)
	phi,y=create_matrices(li_samples,li_labels)
	tm=transition_model.neural_transition_model(model_params)
	em_object=em.em_learner(em_params)
	for iteration in range(em_params['num_iterations']):
		em_object.e_step_m_step(tm,phi,y,iteration)
		if plot==True:
			plot_everything(li_samples,li_labels,tm,phi,ax)
			plt.pause(.5)

