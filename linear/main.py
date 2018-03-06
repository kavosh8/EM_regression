import numpy
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys

def linear_regression(phi,y,w):
	first=lg.inv(numpy.matmul(numpy.matmul(numpy.transpose(phi),w),phi))
	second=numpy.matmul(numpy.matmul(numpy.transpose(phi),w),y)
	theta=first * second
	return theta
def create_train_data(num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(num_samples):
			sample=numpy.random.uniform(-1,1)
			if l==0:
				label=2*sample+1+numpy.random.normal(loc=0.0, scale=0.0,size=1)
			elif l==1:
				label=-2*sample+1+numpy.random.normal(loc=0.0, scale=0.0,size=1)
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
	arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,1)
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels

def compute_posterior(theta1,theta2,phi,y):
	prob1=numpy.array(y-numpy.matmul(phi,theta1))
	prob1=numpy.exp(-numpy.multiply(prob1,prob1)/(2.0))
	prob1_sum=numpy.multiply(prob1,numpy.log(prob1))
	#print(prob1)
	prob2=numpy.array(y-numpy.matmul(phi,theta2))
	prob2=numpy.exp(-numpy.multiply(prob2,prob2)/(2.0))
	prob2_sum=numpy.multiply(prob2,numpy.log(prob2))
	obj=numpy.sum(prob1_sum)+numpy.sum(prob2_sum)
	#print(prob2)
	normalized_prob1=[]
	normalized_prob2=[]
	for index,(p1,p2) in enumerate(zip(prob1,prob2)):
		#print(p1[0],p2[0])
		normalized_prob1.append(p1[0]/(p1[0]+p2[0]))
		normalized_prob2.append(p2[0]/(p1[0]+p2[0]))
		#print(normalized_prob1,normalized_prob2)
		#sys.exit(1)
	normalized_prob1=numpy.matrix(numpy.diag(numpy.array(normalized_prob1)))
	normalized_prob2=numpy.matrix(numpy.diag(numpy.array(normalized_prob2)))
	return normalized_prob1,normalized_prob2,obj
def compute_em_objective():
	print("to be implemented")

num_experiments=200
num_samples=50
num_lines=2
num_iterations=40
fig = plt.figure()
plt.pause(10)
ax1 = fig.add_subplot(1,1,1)
li_obj_all=[]
for experiment in range(num_experiments):
	li_samples,li_labels=create_train_data(num_samples,num_lines)
	phi,y=create_matrices(li_samples,li_labels)
	theta1_a,theta1_b,theta2_a,theta2_b=numpy.random.uniform(-1,1,size=4)

	theta1=numpy.matrix(numpy.array([theta1_a,theta1_b]).reshape(2,1))
	theta2=numpy.matrix(numpy.array([theta2_a,theta2_b]).reshape(2,1))

	li_obj=[]
	plot=True
	for iteration in range(num_iterations):
		#obj=compute_em_objective()
		#sys.exit(1)
		w1,w2,obj=compute_posterior(theta1,theta2,phi,y)
		li_obj.append(obj)
		theta1=linear_regression(phi,y,w1)
		theta2=linear_regression(phi,y,w2)
		#print(obj)

		y1=theta1[0,0]*numpy.array(li_samples)+theta1[1,0]
		y2=theta2[0,0]*numpy.array(li_samples)+theta2[1,0]
		
		if plot==True:
			ax1.clear()
			ax1.plot(li_samples,li_labels,'o')
			ax1.plot(li_samples,y1,lw=2)
			ax1.plot(li_samples,y2,lw=2)
			plt.pause(.05)
	li_obj_all.append(li_obj)

plt.plot(numpy.mean(li_obj_all,axis=0),lw=5)
print(numpy.mean(li_obj_all,axis=0))
plt.ylabel('EM objective')
plt.xlabel('EM iterations')
plt.show()
plt.close()

