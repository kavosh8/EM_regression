import numpy


class em_learner:
	"""docstring for ClassName"""
	def __init__(self, params):
		self.num_iterations=params['num_iterations']
		self.gaussian_variance=params['gaussian_variance']

	def compute_posterior(self,tm,phi,y):
		o_li=tm.predict(phi)
		#print("o_li",o_li)
		y_arr = numpy.squeeze(numpy.asarray(y)).reshape(len(y),1)
		o_li_diff=[]
		for o in o_li:
			o_li_diff.append((o-y_arr).flatten())
		o_li_diff=numpy.transpose(numpy.array(o_li_diff))
		#print("o_li_diff",o_li_diff)
		#o_li_diff=-numpy.multiply(o_li_diff,o_li_diff)/gaussian_variance
		p_arr=numpy.zeros_like(o_li_diff)
		for i in range(o_li_diff.shape[0]):
				
				#print(i,"o_li_diff",o_li_diff[i,:])
				mult=-numpy.multiply(o_li_diff[i,:],o_li_diff[i,:])/self.gaussian_variance
				#print(i,"mult",mult)
				clipped=numpy.clip(mult,a_min=-600, a_max=600)
				#print(i,"clipped",clipped)
				sum_val=numpy.sum(numpy.exp(clipped))
				
				#print(numpy.exp(shifted)/sum_val)
				p_arr[i,:]=numpy.exp(clipped)/sum_val
				#print(i,p_arr[i,:])
		p_arr=numpy.transpose(p_arr)
		#print(p_arr)
		p_li=[]
		for i in range(len(p_arr)):
			p_li.append(numpy.clip(p_arr[i,:],a_min=1e-20, a_max=1))
		#print("p_li",p_li)
		return p_li

	def e_step_m_step(self,tm,phi,y,iteration):
		w_li=self.compute_posterior(tm,phi,y)#E step
		if iteration>0:
			tm.regression(phi,y,w_li)#M step