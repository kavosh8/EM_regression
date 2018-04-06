
import numpy
import matplotlib.pyplot as plt
from rlenv.grid import grid_env as grid_env
import planner

Env = grid_env(True)
print('Initialized, starting to train')
pl=planner.planner()
s = Env.reset()
num_dead=0
for t in range(1000):
	Qs=pl.action_values(s)
	a=pl.choose_action(Qs)
	#a=numpy.random.randint()
	s1,r,dead = Env.step([a])
	plt.pause(0.2)
	Env.plot()
	if dead:
		print(Qs)
		print(a)
		plt.pause(10)
		s = Env.reset() # process smaller batch
		num_dead=num_dead+1
		print("deeeeaaad!!")
	else:
		s=s1
print(num_dead)
    


