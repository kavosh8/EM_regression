import run_pacman, numpy
import planner

planner_parameters=[]
pl=planner.planner(planner_type='stochastic')
run_pacman.run(pl,num_time_steps=1000,show=True)