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
from data_utilities import DataGenerator

# Create Folder List for Folder creation
if __name__ == "__main__":
    import prepare_data_with_valid4 as dataset
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
    batch_size = 10
    lr = 0.0001
    beta1 = 0.9
    n_epoch = 120
    printFreq = 10

    # Define Model Here
    model = UNetWithAttention(shape = (batch_size, nw, nh, nz))

    # Compile Model
    model.compile(optimizer = Adam(lr = lr, beta_1 = beta1), loss = dice_coef_loss, metrics = [dice_coef])

    # Setup DataGenerators
    trainGen = DataGenerator(XTrain, yTrain, batch_size = batch_size)

    # Train the model
    model.fit_generator(trainGen, 
    steps_per_epoch = XTrain.shape[0] // batch_size, 
    epochs = n_epoch, verbose = 1, 
    use_multiprocessing = True, 
    shuffle = True,
    workers = 0)

    # Save the model into json
    SaveModelJSON(model, "unet_attention")
    print("[+] Trained and saved. Please run testing script to load and test")

    if(int(input("[+] To test press 1 : ")) == 1):

        print("[+] Testing")
        # Test the model
        outPred = model.predict(XTest, batch_size = batch_size, verbose=1)

        # Visualize 
        for i in range(outPred.shape[0]):
            VisualizeImageWithPrediction(XTest[i], yTest[i], outPred[i], "./test/all/test_{}.png".format(i))
