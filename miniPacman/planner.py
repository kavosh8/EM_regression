import numpy, sys
import transition_model
import other_models



class planner:
	em_model_object=0
	other_models_object=0


	def __init__(self):
		model_params={}
		model_params['lipschitz_constant']=0.25
		model_params['num_hidden_layers']=1
		model_params['hidden_layer_nodes']=32
		model_params['activation_fn']='relu'
		model_params['learning_rate']=0.001
		model_params['observation_size']=4
		model_params['num_models']=16
		model_params['num_epochs']=5
		model_params['num_samples']=2000
		load=True
		run_ID=0
		gaussian_variance=0.05
		fname='best_models/em_models/model-'+str(run_ID)+"-"+str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
	  	"-"+str(gaussian_variance)+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])
		self.em_model_object=transition_model.neural_transition_model(model_params,load,fname)

		model_params={}
		model_params["observation_size"]=6
		model_params["num_hidden_layers"]=2
		model_params["hidden_layer_nodes"]=32
		model_params["activation_fn"]='relu'
		model_params["learning_rate"]=0.005
		fname='best_models/other_models/'
		self.other_models_object=other_models.neural_other_model(model_params,True,fname)

	def predict(self,state,action):
		action_array=numpy.array(4*[0]).reshape(1,4)
		action_array[0,action]=1
		
		ghost_state=state[2:]
		ghost_state_array=numpy.array(ghost_state).reshape(1,len(ghost_state))

		pacman_state=state[0:2]
		pacman_state_array=numpy.array(pacman_state).reshape(1,len(pacman_state))

		li_ghosts=self.em_model_object.predict(ghost_state_array)
		pacman_next_state=self.other_models_object.pacman_model.predict([pacman_state_array,action_array])
		li_next_states=[]
		li_rewards=[]
		li_dones=[]
		for index,gh in enumerate(li_ghosts):
			next_state=numpy.concatenate((pacman_next_state,gh),axis=1)
			reward=self.other_models_object.reward_model.predict(next_state)
			done=self.other_models_object.done_model.predict(next_state)
			li_next_states.append(next_state[0].tolist())
			li_rewards.append(reward[0,0])
			if done>0.5:
				li_dones.append(True)
			else:
				li_dones.append(False)
		return li_next_states,li_rewards,li_dones

	def action_values(self,state):
		q_li=[]
		for action in range(4):
			li_next_states,li_rewards,li_dones=self.predict(state,action)
			q_li.append(numpy.mean(li_rewards))
		return q_li
	def choose_action(self,Qs,epsilon=0.1):
		if numpy.random.random()>epsilon:
			return numpy.argmax(Qs)
		else:
			return numpy.random.randint(4)


'''
pl=planner()
Qs=pl.action_values([1,1,2,2,3,3])
print(Qs)
a=pl.choose_action(Qs)
print(a)
sys.exit(1)
for s,r,d in zip(li_next_states,li_rewards,li_dones):
	print(s,r,d)
'''


