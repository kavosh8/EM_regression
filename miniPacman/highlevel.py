import run_pacman, numpy
import planner

for planner_type in ['tabular','random','deterministic','stochastic']:
	temp=[]
	pl=planner.planner(planner_type)
	for run in range(20):
		temp.append(run_pacman.run(pl,num_time_steps=1000))
	print(planner_type,numpy.mean(temp))