import numpy, sys
import learn_ghost_model.transition_model
import learn_other_models.other_models


class planner:
	em_model_object=0
	other_models_object=0


	def __init__(self,planner_type):
		self.type=planner_type
		model_params={}
		model_params['lipschitz_constant']=0.3
		model_params['num_hidden_layers']=2
		model_params['hidden_layer_nodes']=32
		model_params['activation_fn']='relu'
		model_params['learning_rate']=0.001
		model_params['observation_size']=2
		model_params['num_models']=4
		model_params['num_epochs']=5
		model_params['num_samples']=49*5
		load=True
		run_ID=3
		gaussian_variance=0.05
		fname='learn_ghost_model/log/model-'+str(run_ID)+"-"+str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
	  	"-"+str(gaussian_variance)+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])
	  	#print('here')
		self.em_model_object=learn_ghost_model.transition_model.neural_transition_model(model_params,load,fname)

		model_params={}
		model_params["observation_size"]=6
		model_params["num_hidden_layers"]=2
		model_params["hidden_layer_nodes"]=32
		model_params["activation_fn"]='relu'
		model_params["learning_rate"]=0.005
		fname='learn_other_models/'
		self.other_models_object=learn_other_models.other_models.neural_other_model(model_params,True,fname)

	def predict_ghost_next_states(self,ghosts):
		ghost1=numpy.array(ghosts[:2]).reshape(1,2)
		ghost2=numpy.array(ghosts[2:4]).reshape(1,2)
		t1=self.em_model_object.predict(ghost1)
		t2=self.em_model_object.predict(ghost2)
		li_location=[]
		li_probs=[]
		probs=self.em_model_object.probs
		for index1,x in enumerate(t1):
			for index2,y in enumerate(t2):
				temp=x[0].tolist()+y[0].tolist()
				li_location.append(temp)
				li_probs.append(probs[index1]*probs[index2])
		return li_location,li_probs

	def predict(self,state,action):
		#state=[0,0,0,1,6,6]
		ghost_next_locs,ghost_next_probs=self.predict_ghost_next_states(state[2:])
		action_array=numpy.array(4*[0]).reshape(1,4)
		action_array[0,action]=1
		pacman_state_array=numpy.array(state[:2]).reshape(1,2)
		pacman_next_loc=self.other_models_object.pacman_model.predict([pacman_state_array,action_array])

		next_states=[pacman_next_loc[0].tolist()+ gnl for gnl in ghost_next_locs]
		next_rewards=self.other_models_object.reward_model.predict(next_states)
		next_rewards=next_rewards.tolist()

		Q=numpy.mean([nr[0]*gnp for (nr,gnp) in zip(next_rewards,ghost_next_probs)])
		return Q

	def action_values(self,state):
		q_li=[]
		for action in range(4):
			Q=self.predict(state,action)
			q_li.append(Q)
		return q_li
		
	def choose_action(self,s,epsilon=0):
		if self.type=='random':
			return numpy.random.randint(4)
		elif self.type=='stochastic':
			Qs=self.action_values(s)
			if numpy.max(Qs)==numpy.min(Qs):
				return numpy.random.randint(4)
			if numpy.random.random()>epsilon:
				#print(Qs)
				return numpy.argmax(Qs)
			else:
				return numpy.random.randint(4)


