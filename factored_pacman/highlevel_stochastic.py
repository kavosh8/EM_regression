import run_pacman, numpy
import planner
import sys

model_params={}
model_params['lipschitz_constant']=float(sys.argv[1])
model_params['run_ID']=int(sys.argv[2])
model_params['learning_rate']=float(sys.argv[3])
model_params['gaussian_variance']=float(sys.argv[4])


pl=planner.planner(planner_type='stochastic',model_params=model_params)
temp=[]
for _ in range(100):
	temp.append(run_pacman.run(pl,num_time_steps=1000,show=False))
	numpy.savetxt('returns/'+str(model_params['lipschitz_constant'])+"-"+str(model_params['run_ID'])+"-"+str(model_params['learning_rate'])+"-"+str(model_params['gaussian_variance'])+".txt",temp)