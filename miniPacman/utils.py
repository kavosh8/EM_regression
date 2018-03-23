import numpy, scipy

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

def load_synthetic_data(N):
	li_s,li_sprime=[],[]
	for _ in range(N):
		x1,y1,x2,y2=numpy.random.uniform(0,10,4)
		s=[x1,y1,x2,y2]
		case=numpy.random.randint(0,16)
		if case==0:
			s_p=[x1+1,y1,x2+1,y2]
		elif case==1:
			s_p=[x1+1,y1,x2-1,y2]
		elif case==2:
			s_p=[x1+1,y1,x2,y2+1]
		elif case==3:
			s_p=[x1+1,y1,x2,y2-1]
		elif case==4:
			s_p=[x1-1,y1,x2+1,y2]
		elif case==5:
			s_p=[x1-1,y1,x2-1,y2]
		elif case==6:
			s_p=[x1-1,y1,x2,y2+1]
		elif case==7:
			s_p=[x1-1,y1,x2,y2-1]
		if case==8:
			s_p=[x1,y1+1,x2+1,y2]
		elif case==9:
			s_p=[x1,y1+1,x2-1,y2]
		elif case==10:
			s_p=[x1,y1+1,x2,y2+1]
		elif case==11:
			s_p=[x1,y1+1,x2,y2-1]
		elif case==12:
			s_p=[x1,y1-1,x2+1,y2]
		elif case==13:
			s_p=[x1,y1-1,x2-1,y2]
		elif case==14:
			s_p=[x1,y1-1,x2,y2+1]
		elif case==15:
			s_p=[x1,y1-1,x2,y2-1]
		li_s.append(s)
		li_sprime.append(s_p)
	return li_s,li_sprime

def create_matrices(li_samples,li_labels,model_params):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,model_params['observation_size'])
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,model_params['observation_size'])
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels
def compute_approx_wass_loss(tm,em_object):

	w_loss=0
	N=500
	for _ in range(N):
		sample_li=numpy.random.uniform(0,10,4)
		estimated_labels=[m.predict(numpy.array(sample_li).reshape(1,4))[0] for m in tm.models]
		estimated_labels=numpy.array(estimated_labels)
		for i in range(estimated_labels.shape[1]):
			estimated_labels_column=estimated_labels[:,i].tolist()
			estimated_labels_probs=em_object.learned_priors
			#print(estimated_labels_column,estimated_labels_probs)
			true_labels=[sample_li[i]]*8+([sample_li[i]+1])*4+([sample_li[i]-1])*4
			true_probs=16*[(1./16)]
			#print(true_labels,true_probs)
			w_loss=w_loss+wasserstein(true_probs,true_labels,estimated_labels_probs,estimated_labels_column)
	return w_loss/N