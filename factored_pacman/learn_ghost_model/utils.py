import numpy, scipy, sys, csv

def wasserstein(true_probs,true_labels,estimated_probs,estimated_labels):
	W=scipy.stats.wasserstein_distance(true_labels, estimated_labels, u_weights=true_probs, v_weights=estimated_probs)
	return W

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

def state_2_number(state):
	print(state)
	if len(state)!=2:
		print("wow")
		sys.exit(1)
	return state[0] + state[1]*7

def number_2_state(number):
	li=[]
	for _ in range(2):
		temp=int(number%7)
		li.append(temp)
		number=number/7
	li.append(number)
	return li