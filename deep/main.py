import numpy
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import net

def linear_regression(phi,y,w):
	first=lg.inv(numpy.matmul(numpy.matmul(numpy.transpose(phi),w),phi))
	second=numpy.matmul(numpy.matmul(numpy.transpose(phi),w),y)
	theta=first * second
	return theta
def deep_regression(phi,y,net,w):
	net.fit(x=phi,y=y,epochs=100, verbose=1,sample_weight=w)
	return net
	#sys.exit(1)
def create_train_data(num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(num_samples):
			sample=numpy.random.uniform(-1,1)
			if l==0:
				label=numpy.sin(3*sample)+numpy.random.normal(loc=0.0, scale=0.0,size=1)
			elif l==1:
				label=numpy.cos(3*sample)+numpy.random.normal(loc=0.0, scale=0.0,size=1)
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

def compute_posterior(net1,net2,phi,y):
	out1=net1.predict(phi)
	out2=net2.predict(phi)

	prob1=numpy.array(y-out1)
	prob1=numpy.exp(-numpy.multiply(prob1,prob1)/(.05))
	prob1_sum=numpy.multiply(prob1,numpy.log(prob1))


	prob2=numpy.array(y-out2)
	prob2=numpy.exp(-numpy.multiply(prob2,prob2)/(.05))
	prob2_sum=numpy.multiply(prob2,numpy.log(prob2))
	obj=numpy.sum(prob1_sum)+numpy.sum(prob2_sum)
	#print(prob2)
	#sys.exit(1)
	normalized_prob1=[]
	normalized_prob2=[]
	for index,(p1,p2) in enumerate(zip(prob1,prob2)):
		#print(p1[0],p2[0])
		normalized_prob1.append(p1[0]/(p1[0]+p2[0]))
		normalized_prob2.append(p2[0]/(p1[0]+p2[0]))
		#print(normalized_prob1,normalized_prob2)
		#sys.exit(1)
	normalized_prob1=numpy.array(normalized_prob1)
	normalized_prob2=numpy.array(normalized_prob2)
	return normalized_prob1,normalized_prob2,obj
def compute_em_objective():
	print("to be implemented")

num_experiments=1
num_samples=20
num_lines=2
num_iterations=100
fig = plt.figure()
#plt.pause(10)
ax1 = fig.add_subplot(1,1,1)
li_obj_all=[]
for experiment in range(num_experiments):
	li_samples,li_labels=create_train_data(num_samples,num_lines)
	phi,y=create_matrices(li_samples,li_labels)
	net1=net.create_model()
	net2=net.create_model()
	li_obj=[]
	plot=True
	#plt.pause(10)
	for iteration in range(num_iterations):
		#obj=compute_em_objective()
		#sys.exit(1)
		w1,w2,obj=compute_posterior(net1,net2,phi,y)
		li_obj.append(obj)
		if iteration>0:
			net1=deep_regression(phi,y,net1,w1)
			net2=deep_regression(phi,y,net2,w2)
		y1=net1.predict(phi)
		y2=net2.predict(phi)
		#print(y1)
		#sys.exit(1)
		if plot==True:
			ax1.clear()
			ax1.plot(li_samples,li_labels,'o')
			ax1.plot(li_samples,y1,'o',lw=2)
			ax1.plot(li_samples,y2,'o',lw=2)

			plt.pause(.05)
		#sys.exit(1)
	li_obj_all.append(li_obj)

plt.plot(numpy.mean(li_obj_all,axis=0),lw=5)
print(numpy.mean(li_obj_all,axis=0))
plt.ylabel('EM objective')
plt.xlabel('EM iterations')
plt.show()
plt.close()

