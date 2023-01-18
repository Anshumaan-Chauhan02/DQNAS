# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:40:18 2022

@author: AnshumaanChauhan
"""

import CNNCONSTANTS
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from NASutils import *
from cnnnas import CNNNAS
from CNNCONSTANTS import TOP_N
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#Use MNIST for the time being 
#data = pd.read_csv('DATASETS/wine-quality.csv')
#x = data.drop('quality_label', axis=1, inplace=False).values
#y = pd.get_dummies(data['quality_label']).values

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
train_y=to_categorical(train_y,num_classes=10)
train_new_X=train_X[:2]
train_new_y=train_y[:2]
nas_object = CNNNAS(train_new_X, train_new_y)
data = nas_object.search()
get_top_n_architectures(TOP_N)