
import numpy
import matplotlib.pyplot as plt
from rlenv.grid import grid_env as grid_env

def run(pl,num_time_steps):
	show=False
	Env = grid_env(show)
	#print('Initialized, starting to train')
	
	s = Env.reset()
	num_dead=0
	for t in range(num_time_steps):
		a=pl.choose_action(s)
		#print(a)
		#a=numpy.random.randint()
		s1,r,dead = Env.step([a])
		if show:
			plt.pause(0.05)
			Env.plot()
		if dead:
			s = Env.reset() 
			num_dead=num_dead+1
			#print("deeeeaaad!!")
		else:
			s=s1
	print(num_dead)
	return num_dead
    


