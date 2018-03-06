import numpy 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

k=.4
def create_model():
	model=Sequential()
	model.add(Dense(16, input_dim=1, activation='relu',kernel_initializer='random_uniform',W_constraint = WeightClip(k)))
	#model.add(Dense(16, input_dim=1, activation='tanh',kernel_initializer='random_uniform',W_constraint = WeightClip(2.5)))
	model.add(Dense(16, input_dim=1, activation='relu',kernel_initializer='random_uniform',W_constraint = WeightClip(k)))
	model.add(Dense(1))
	ad=optimizers.Adam(lr=0.01)
	model.compile(loss='mean_squared_error',optimizer=ad)
	return model