import run_pacman, numpy
import planner

for planner_type in ['stochastic','random']:
	temp=[]
	pl=planner.planner(planner_type)
	for run in range(50):
		temp.append(run_pacman.run(pl,num_time_steps=1000,show=False))
		numpy.savetxt("returns/"+planner_type+".txt", temp)
		print(planner_type,numpy.mean(temp))