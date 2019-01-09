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
from model import *
import prepare_data_with_valid4 as dataset

# Create Folder List for Folder creation
folderList = ["./checkpoint", 
"./sample", 
"./sample/all",
"./test",
"./test/all"]

InitializeFolders(folderList)

# Set up data
XTrain = dataset.X_train_input
yTrain = dataset.X_train_target[:, :, :, np.newaxis]
XTest = dataset.X_dev_input
yTest = dataset.X_dev_target[:, :, :, np.newaxis]

yTrain = (yTrain > 0).astype(int)
yTest = (yTest > 0).astype(int)
X = np.asarray(XTrain[80])
y = np.asarray(yTrain[80])
nw, nh, nz = X.shape

# Hyperparameter settings
batch_size = 5
lr = 0.0001
beta1 = 0.9
n_epoch = 5
printFreq = 10

# Define Model Here
model = UNet(shape = (batch_size, nw, nh, nz))

# Compile Model
model.compile(optimizer = Adam(lr = lr, beta_1 = beta1), loss = jaccard_distance_loss, metrics = ['accuracy'])

# Train the model
train_history = History()
for epoch in range(0, n_epoch):
    n_batch = 0
    print("[+] Epoch ==> ",epoch + 1)
    for batch in tl.iterate.minibatches(inputs = XTrain, targets = yTrain, batch_size = batch_size, shuffle = True):
        images, labels = batch

        data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                        images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                        images[:,:,:,3, np.newaxis], labels)], fn = DistortImages)
        bImages = data[:, 0:4, :, :, :]
        bLabels = data[:, 4, :, :, :]
        bImages = bImages.transpose((0, 2, 3, 1, 4))
        bImages.shape = (batch_size, nw, nh, nz)

        model.fit(x = bImages, y = labels, batch_size = batch_size, verbose = 0, shuffle = False, callbacks=[train_history])
        n_batch += 1
        
        if n_batch % printFreq == 0:
            print("                                                                                                                          ", end = "\r")
            print("batch ==> ", n_batch, ", images parsed ==> ", batch_size * n_batch, end = "")
            print("loss: ", train_history.history["loss"], "accuracy: ", train_history.history["acc"], end = "\r")
    
    print("[+] Epoch over ==> ", epoch + 1, " out of ", n_epoch)
    print(train_history.history)

# Save the model into json
SaveModelJSON(model, "unet_standard")
print("[+] Trained and saved. Please run testing script to load and test")

if(int(input("[+] To test press 1 : ")) == 1):

    print("[+] Testing")
    # Test the model
    outPred = model.predict(XTest, batch_size = batch_size, verbose=1)

    # Visualize 
    for i in range(outPred.shape[0]):
        VisualizeImageWithPrediction(XTest[i], yTest[i], outPred[i], "./test/all/test_{}.png".format(i))
