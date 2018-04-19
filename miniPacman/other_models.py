import numpy 
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
from keras.constraints import Constraint
from keras import backend as K
import sys

class neural_other_model:
    model=0
    def __init__(self,model_params,load=False,fname='empty'):
        self.observation_size=model_params["observation_size"]
        self.num_hidden_layers=model_params["num_hidden_layers"]
        self.hidden_layer_nodes=model_params["hidden_layer_nodes"]
        self.activation_fn=model_params["activation_fn"]
        self.learning_rate=model_params["learning_rate"]
        self.reward_model=self.create_reward_model()
        self.done_model=self.create_done_model()
        self.pacman_model=self.create_pacman_model()
        self.ghosts_model=self.create_ghosts_model()
        if load==True:
            self.load_model(fname)
            self.ghosts_tabular_model=numpy.loadtxt('best_models/tabular_models/ghosts.h5')
            print(self.ghosts_tabular_model)
            #sys.exit(1)

    def load_model(self,fname):
        self.reward_model.load_weights(fname+"reward.h5")
        self.done_model.load_weights(fname+"done.h5")
        self.pacman_model.load_weights(fname+"pacman.h5")
        self.ghosts_model.load_weights(fname+"ghosts.h5")

    def create_done_model(self):
            input_state = keras.layers.Input(shape=(self.observation_size,))
            h=input_state
            for l in range(self.num_hidden_layers):
                h=Dense(self.hidden_layer_nodes
                        ,activation=self.activation_fn)(h)

            output_reward=Dense(1,activation='sigmoid')(h)
            model=keras.models.Model(inputs=input_state, outputs=output_reward)
            ad=optimizers.Adam(lr=self.learning_rate)
            model.compile(loss='binary_crossentropy',optimizer=ad)
            return model

    def create_reward_model(self):
        input_state = keras.layers.Input(shape=(self.observation_size,))
        h=input_state
        for l in range(self.num_hidden_layers):
            h=Dense(self.hidden_layer_nodes
                    ,activation=self.activation_fn)(h)

        output_reward=Dense(1)(h)
        model=keras.models.Model(inputs=input_state, outputs=output_reward)
        ad=optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error',optimizer=ad)
        return model
    def create_pacman_model(self):
        pacman_size=2
        input_state = keras.layers.Input(shape=(pacman_size,))
        input_action = keras.layers.Input(shape=(4,))
        h=keras.layers.Concatenate(axis=-1)([input_state,input_action])
        for l in range(self.num_hidden_layers):
            h=Dense(self.hidden_layer_nodes
                    ,activation=self.activation_fn)(h)
        output_pacman=Dense(pacman_size)(h)
        model=keras.models.Model(inputs=[input_state,input_action], outputs=output_pacman)
        ad=optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error',optimizer=ad)
        return model
    def create_ghosts_model(self):
        input_size=4
        input_state = keras.layers.Input(shape=(input_size,))
        h=input_state
        for l in range(self.num_hidden_layers):
            h=Dense(self.hidden_layer_nodes
                    ,activation=self.activation_fn)(h)
        out=Dense(input_size)(h)
        model=keras.models.Model(inputs=input_state,outputs=out)
        ad=optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error',optimizer=ad)
        return model




