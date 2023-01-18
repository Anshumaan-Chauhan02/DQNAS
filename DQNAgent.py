# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:42:31 2022

@author: AnshumaanChauhan
"""
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout, Conv2d, MaxPool2d, Activation, Flatten, Input
from tf.keras.callbacks import TensorBoard
from tf.keras.activations import relu
from tf.keras.optimizers import Adam
from collections import deque
import time 
import random
import numpy as np

REPLAY_MEMORY_SIZE = 50000 # Can also write as 50_000 for readability 
MODEL_NAME= "First_Try"
MIN_REPLAY_MEMORY_SIZE=1000
MINIBATCH_SIZE= 64
DISCOUNT= 0.99
UPDATE_TARGET_EVERY=5   


class ModifiedTensorBoard(TensorBoard):
    
    #By default Keras wants to create a new TensorBoard file after every fit 
    #But we want only a single log file, therefore, this class is created to solve this issue 
    
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    
    def __init__(self):
        
        #Main model
        self.model=self.create_model()
        #Target model
        self.target_model=self.create_model()
        self.target_model.set_weights(self.model.get_weights()) 
        
        #Initially it will do just random exploration and eventually learn about the optimal value
        #Therefore, not advisable to update weights after each predict 
        
        #Will fit main model that will fitted after every step (Trained every step)
        #Target model will be the one we will do predict every step
        
        #After some n number of epochs we set weights of Train model same as that of Main model 
        #Stablises the model, and a lot of randomness is noticed in initial steps 
        
        self.replay_memory= deque(maxlen=REPLAY_MEMORY_SIZE)
        #List with a fixed max length , will store last maxlen number of steps of Main model 
        
        #Batch Learning generally makes a better and stabilised model (doesn't overfits)
        #Now we take a random samle out of these 50000 memory and then this batch is what we feed to Target Model
        
        self.tensorBoard= ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        
        self.target_update_counter= 0 # Will use to track and tell when to update the Target Network 
        
     
        
    def create_model():
        model= Sequential()
        model.add(Input(16,)) #We have to check about the input states (Observation states) #Number of max layers (in terms of layer id)
        model.add(Dense(32,activation='relu')) 
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='linear')) #Output  is number of Action state space - Number of possible actions, like vocab size
        
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics= ['accuracy'])     
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        #Transition is (Observation space , action,  reward,  new observatoin state)
    
        #Main model get the q values of all possible actions for the current state
    def get_qs(self, state, step):
        return self.model.predict(state)[0]
    #Return a one element array
    
    #Only train when certain number of samples have been stord in the replay table
    def train(self, terminal_state, step):
        if len(self.replay_memory)< MIN_REPLAY_MEMORY_SIZE:
            return 
        
        minibatch=random.sample(self.replay_memory,MINIBATCH_SIZE)
        
        
        current_states= np.array([transition[0] for transition in minibatch])
        current_qs_list= self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        
        future_qs_list=self.target_model.predict(new_current_states)
        #Need Q values for future current states in order to apply the formula for updation of Q values
        
        X=[] #Images from gaem, what will be the input
        y=[] #Action we take, what are the action, Input might be the model and output might be the accuracy predicted
        
        #Done is whether we are done with the environment or not 
        #rest 3 are what is present in minibatch 
        #Used to caluclate the second half of the updation formula for Q-values
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch) :
            if not done: 
                max_future_q= np.max(future_qs_list[index])
                new_q=reward+ DISCOUNT*max_future_q
            if done :
                new_q =reward
                
            current_qs=current_qs_list[index]
            current_qs[action]=new_q
            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X),np.array(y), batch_size= MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorBoard] if terminal_state else None)
    #Output of the Neural Network is the Q-values, so in order to update the Q-value (max one) generated,
    #we save the output, make changes in it, and then fit the neural network that it generates our specified values
    #If on terminal state do fit else do nothing 

        #Updating thecounter and checking whether we want to update the model or not
        if terminal_state:
            self.target_update_counter+=1
            
        if self.target_update_counter>UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter=0
            
        