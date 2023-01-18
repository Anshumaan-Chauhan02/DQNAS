# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:58:50 2022

@author: AnshumaanChauhan
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input, LSTM
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.activations import relu
from keras.optimizers import *
from collections import deque
import time 
import random
import os
import tensorflow.keras.optimizers
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import CNNCONSTANTS
from CNNCONSTANTS import *

from CNNGenerator import CNNSearchSpace


from operator import itemgetter

#In order to make results comparable for different models 
random.seed(1)


class DQNAgent (CNNSearchSpace):
    
    def __init__(self):
        
        REPLAY_MEMORY_SIZE = 50000 # Can also write as 50_000 for readability 
        MODEL_NAME= "First_Try"
        self.MIN_REPLAY_MEMORY_SIZE=10
        #MINIBATCH_SIZE= 64 #Usually the size of mini batch is 32, 64, or multiple of 8
        #DISCOUNT= 0.99
        self.UPDATE_TARGET_EVERY=3


        #Constants for epsilon greedy algo
        self.EPSILON=1 #Will be decayed over the training process
        self.EPSILON_DECAY=0.01
        self.seq_data=[]
        self.replay_memory=[]
        #self.replay_memory= deque(maxlen=REPLAY_MEMORY_SIZE)
        #Will store x,y, val accuracy, pred_accuracy for all the things geenrated
        #List with a fixed max length , will store last maxlen number of steps of Main model 
        
        #Batch Learning generally makes a better and stabilised model (doesn't overfits)
        #Now we take a random samle out of these 50000 memory and then this batch is what we feed to Target Model
        
        self.target_update_counter= 0 # Will use to track and tell when to update the Target Network 
        
        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR
        
        # inheriting from the search space
        super().__init__(TARGET_CLASSES)

        # number of classes for the controller (+ 1 for padding)
        self.controller_classes = len(self.vocab) + 1
        
        # file path of controller weights to be stored at
        self.controller_weights = 'LOGS2/controller_weights.h5'
        
        #Main model
        #self.model=self.create_control_model()
        #Target model
        #self.target_model=self.create_control_model()
        #self.target_model.set_weights(self.model.get_weights()) 
        
        #Initially it will do just random exploration and eventually learn about the optimal value
        #Therefore, not advisable to update weights after each predict 
        
        #Will fit main model that will fitted after every step (Trained every step)
        #Target model will be the one we will do predict every step
        
        #After some n number of epochs we set weights of Train model same as that of Main model 
        #Stablises the model, and a lot of randomness is noticed in initial steps 
        
    def sample_architecture_sequences(self, model, number_of_samples):
        # define values needed for sampling 
        final_layer_id = len(self.vocab)
        BatchNorm_id = final_layer_id - 1
        Flatten_id=final_layer_id-2
        vocab_idx = [0] + list(self.vocab.keys())
        
        # initialize list for architecture samples
        samples = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        
        # while number of architectures sampled is less than required
        while len(samples) < number_of_samples:
            
            # initialise the empty list for architecture sequence
            seed = []
            
            # while len of generated sequence is less than maximum architecture length
            while len(seed) < self.max_len:
                
                # pad sequence for correctly shaped input for controller
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)
                
                # given the previous elements, get softmax distribution for the next element
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                #print(probab[0])
                #print(len(probab[0]))
                #probab = probab[0][0]
                #print(probab)
                # sample the next element randomly given the probability of next elements (the softmax distribution)
                
                '''
                random_val=random.random()
                if self.EPSILON > 0:
                    if random_val<self.EPSILON:
                        next = np.random.choice(vocab_idx, size=1, p=probab[0])
                        next=next[0]
                        self.EPSILON=self.EPSILON-self.EPSILON_DECAY
                    else:
                        best_action= max(probab[0])
                        list_rep=probab[0].tolist()
                        next=vocab_idx[list_rep.index(best_action)]
                else:
                    best_action= max(probab[0])
                    list_rep=probab[0].tolist()
                    next=vocab_idx[list_rep.index(best_action)]
                '''
                
                next = np.random.choice(vocab_idx, size=1, p=probab[0])
                #Here we have to specify a range of values, to cover the point of dropout cannot be the first layer 
                if (next >= self.conv_id) and len(seed) == 0:
                    continue
                #Have to make a rule such that first layer cannot be anything except the Convolutional Layer
                
                # first layer is not final layer
                if next == final_layer_id and len(seed) == 0:
                    continue
                
                # if final layer, break out of inner loop
                if next == final_layer_id:
                    seed.pop()
                    seed.append(Flatten_id)
                    seed.append(next)
                    break
                
                # if sequence length is 1 less than maximum, add final
                # layer and break out of inner loop
                if len(seed) == self.max_len - 2:
                    seed.append(Flatten_id)
                    seed.append(final_layer_id)
                    break
                
                # ignore padding
                if not next == 0:
                    check_insert=False
                    if next > self.pool_id and next <= self.fully_id :
                        for i in seed:
                            if i == Flatten_id :
                                seed.append(next)
                                check_insert=True
                                break
                    
                    if next == Flatten_id :
                        check_dupli_flatten=False
                        for i in seed :
                            if i==next:
                                check_dupli_flatten=True
                        if not check_dupli_flatten:
                            seed.append(next)
                            check_insert=True
                    
                    if next <= self.pool_id :
                        check_no_pool_conv_after_flatten=False
                        for i in seed: 
                            if i == Flatten_id:
                                check_no_pool_conv_after_flatten=True
                        if not check_no_pool_conv_after_flatten :
                            seed.append(next)
                            check_insert=True
                    
                    if next > self.fully_id and next<=self.reg_layer_id:
                        i=seed[-1]
                        if ((i>self.conv_id and i<=self.pool_id) or (i>self.pool_id and i<=self.fully_id)):
                            seed.append(next)
                            check_insert=True
                    
                    if next== BatchNorm_id:
                        i=seed[-1]
                        if i>self.fully_id and i<=self.reg_layer_id:
                            seed.append(next)
                            check_insert=True
                    
                    if not check_insert:
                        seed.append(next)
                else:
                    continue
            # check if the generated sequence has been generated before.
            # if not, add it to the sequence data. 
            if seed not in self.seq_data:   
                samples.append(seed)
                self.seq_data.append(seed)
        return samples    
    
    def create_control_model(self, controller_input_shape, controller_batch_size):
        
        main_input=Input(shape=controller_input_shape, name='main_input')
        x= LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        x2=Dropout(0.2)(x)
        x3=LSTM(self.controller_lstm_dim)(x2)
        x4=Dropout(0.2)(x3)
        main_output=Dense(self.controller_classes, activation='softmax', name='main_output')(x4)
        model=Model(inputs=[main_input],outputs=[main_output])
        return model
    
    
    def create_hybrid_model(self, controller_input_shape, controller_batch_size):
        
        main_input=Input(shape=controller_input_shape, name='main_input')
        x= LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        x2=Dropout(0.2)(x)
        x3=LSTM(self.controller_lstm_dim)(x2)
        x4=Dropout(0.2)(x3)
        main_output=Dense(self.controller_classes, activation='softmax', name='main_output')(x4)

        # LSTM layer
        x5 = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        # single neuron sigmoid layer for accuracy prediction
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x2)
    
        # finally the Keras Model class is used to create a multi-output model
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model
    
    
    #Only train when certain number of samples have been stord in the replay table
    def train_control_model(self, model, target_model, x_data, y_data, val_accuracy, loss_func, controller_batch_size, nb_epochs):
        

        for i in range(len(x_data)):
            self.replay_memory.append([x_data[i][0],y_data[i],val_accuracy[i]])
        
        if len(self.replay_memory)<self.MIN_REPLAY_MEMORY_SIZE:
            return
        #Top 250 Architectures are taken
        self.replay_memory= sorted(self.replay_memory,key=itemgetter(2))
        to_train=self.replay_memory[:1]
        optim = getattr(tensorflow.keras.optimizers, self.controller_optimizer)(lr=self.controller_lr, 
                                                       decay=self.controller_decay)
                                                       
        # compile model depending on loss function and optimizer provided
        model.compile(optimizer=optim, loss={'main_output': loss_func})
        
        # load controller weights
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
        
        x_data=[]
        y_data=[]
        for i in range(len(to_train)):
            x_data.append(to_train[i][0].reshape(1,7))
            y_data.append(to_train[i][1])
        
        # train the controller
        
        #We are trying to make it learn that if the previous layers are given in this order and the next predicted is final 
        #Taking the ones with best accuracy helps it learn which is better 
        print("TRAINING CONTROLLER...")
        model.fit({'main_input': np.array(x_data)},
                  {'main_output': np.array(y_data)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        #{'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)}
        # save controller weights
        model.save_weights(self.controller_weights)

        #Updating the counter and checking whether we want to update the model or not
        self.target_update_counter+=1
            
        #If we are at the point where counter value id reached, then we just copy the weights from Main model to Target Model     
        if self.target_update_counter>self.UPDATE_TARGET_EVERY:
            print("TRANSFERRING WEIGHTS...")
            target_model.set_weights(model.get_weights())
            #Reinitialize target update counter value to 0 
            self.target_update_counter=0
            
    

    def train_hybrid_model(self, model, target_model, x_data, y_data, val_accuracy, pred_accuracy, loss_func, controller_batch_size, nb_epochs):
        
        for i in range(len(x_data)):
            self.replay_memory.append([x_data[i][0],y_data[i],val_accuracy[i],pred_accuracy[i]])
        
        if len(self.replay_memory)<self.MIN_REPLAY_MEMORY_SIZE:
            return
        #Top 250 Architectures are taken
        self.replay_memory= sorted(self.replay_memory,key=itemgetter(2))
        to_train=self.replay_memory[:1]
        optim = getattr(tensorflow.keras.optimizers, self.controller_optimizer)(lr=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        
        
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1})
        
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
        
        x_data=[]
        y_data=[]
        pred_target=[]
        for i in range(len(to_train)):
            x_data.append(to_train[i][0].reshape(1,7))
            y_data.append(to_train[i][1])
            pred_target.append(to_train[i][3])
        print("TRAINING CONTROLLER...")
        model.fit({'main_input': np.array(x_data)},
                  {'main_output': np.array(y_data),
                   'predictor_output': np.array(pred_target)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        model.save_weights(self.controller_weights)
        
        self.target_update_counter+=1
            
        #If we are at the point where counter value id reached, then we just copy the weights from Main model to Target Model     
        if self.target_update_counter>self.UPDATE_TARGET_EVERY:
            print("TRANSFERRING WEIGHTS...")
            self.target_model.set_weights(model.get_weights())
            #Reinitialize target update counter value to 0 
            self.target_update_counter=0


    
    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        pred_accuracies = []        
        for seq in seqs:
            # pad each sequence
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            # get predicted accuracies
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies
    