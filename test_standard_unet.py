import numpy as np
import tensorflow as tf
import tensorlayer as tl
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from keras.losses import *
from keras import backend as K
import os, time
from matplotlib import pyplot as plt
from model_utilities import *
import prepare_data_with_valid4 as dataset

# Prepare dataset
XTest = dataset.X_dev_input
yTest = dataset.X_dev_target[:, :, :, np.newaxis]

yTest = (yTest > 0).astype(int)

# Hyperparameter settings
batch_size = 5
lr = 0.0001
beta1 = 0.9
n_epoch = 5
printFreq = 10

# Load JSON Model
model = LoadModelJSON("unet_standard")

# Compile model
model.compile(optimizer = Adam(lr = lr, beta_1 = beta1), loss = jaccard_distance_loss, metrics = ['accuracy'])

# Test the model
outPred = model.predict(XTest, batch_size = batch_size, verbose=1)

# Visualize 
for i in range(outPred.shape[0]):
    VisualizeImageWithPrediction(XTest[i], yTest[i], outPred[i], "./test/all/test_{}.png".format(i))
