import numpy as np
import os, time, logging, traceback
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import tensorflow as tf
import tensorlayer as tl
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from error_handling_utilities import ModelFrameError
from error_handling_utilities import logClass as log
from DataGenerator import DataGenerator
from data_utilities import VisualizeImage, VisualizeImageWithPrediction, DistortImages
from model_utilities import LoadModelJSON, SaveModelJSON

class UNet:

    # NOTE: Private functions
    def __init__(self, shape:tuple, n_out = 1, dropout = 0.5, batchNorm = False, modelPath = None, modelName = None, loadModel = False):
        '''
        initialize the UNet model
        
        Arguments:
            shape {tuple of integer} -- the shape describing the input shape
        
        Keyword Arguments:
            n_out {int} -- the output size (default: {1})
            dropout {float} -- dropout prob. (default: {0.5})
            batchNorm {bool} -- status check to do batch normalization (default: {False})            
            modelPath {str} -- path of the saved model to load (default: {None})
            modelName {str} -- the name of the model to load (default: {None})
            loadModel {bool} -- status check for loading model(default: {False})
        '''
        # Setup model
        self.model = None
        if loadModel == True:
            self.model = self.LoadModel(path = modelPath, modelName = modelName)
            print("[INFO] Model loaded. Compile it")
            log.logger.info("[INFO] Model Loaded. Compile it!")

        else:
            self.model = self.__InitializeModel__(shape = shape, n_out = n_out, dropout = dropout, batchNorm = batchNorm)
            print("[INFO] Model built. Compile it!")
            log.logger.info("[INFO] Model built. Compile it!")

        # Setup folder list
        folderList = ["./checkpoint", 
        "./test",
        "./savedModels",
        "./plots"]
        self.__InitializeFolders__(folderList)
    
    def __InitializeFolders__(self, folderList: list) -> bool:
        '''
        Function to initialize all the required folders for the model to
        run. NOTE: PRIVATE FUNCTION -- DON'T USE
        
        Arguments:
            folderList {list} -- Consist of all the folders to initialize
        
        Returns:
            bool -- status return
        '''
        for folder in folderList:
            try:
                if not os.path.exists(folder):
                    os.mkdir(folder)
            except Exception as err:
                print("[FATAL] Exception occured while making folders: ", err)
                traceback.print_exc()
                log.logger.critical("[FATAL] Exception occured while making folders: " + err)
                return False
        print("[INFO] Folders initialized")
        log.logger.info("[INFO] Folders initialized")
        return True

    def __Conv2DBlock__(self, inputTensor, filters:int, kernelSize = 3, batchNorm = False):
        '''
        A conv2D block that defines a full 2 layer conv2d with batchnorm 
        # NOTE: PRIVATE FUNCTION -- DON'T USE

        Arguments:
            inputTensor {tensor} -- the input tensor
            filters {int} -- the total no. of filters

        Keyword Arguments:
            kernelSize {int} -- the size of each filter (default: {3})
            batchNorm {bool} -- status check for batchnorm (default: {False})
        '''
        # First layer
        x = Conv2D(filters = filters, kernel_size = (kernelSize, kernelSize), kernel_initializer = "he_normal", padding = "same")(inputTensor)
        if batchNorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Second layer
        x = Conv2D(filters = filters, kernel_size = (kernelSize, kernelSize), kernel_initializer = "he_normal", padding = "same")(x)
        if batchNorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def __InitializeModel__(self, shape, n_out = 1, dropout = 0.5, batchNorm = False, ) -> Model:
        '''
        Function to initialize the UNet code # NOTE: PRIVATE FUNCTION -- DON'T USE
        
        Arguments:
            shape {tuple of int} -- shape of the input
        
        Keyword Arguments:
            n_out {int} -- the output size (default: {1})
            dropout {float} -- dropout prob. (default: {0.5})
            batchNorm {bool} -- status check to do batch normalization (default: {False})
        
        Returns:
            Model -- the builded model -- YET TO COMPILE
        '''

        nnx = int(shape[1])
        nny = int(shape[2])
        nnz = int(shape[3])
        print(" * Input: size of image: %d %d %d" % (nnx, nny, nnz))

        # Encoder 
        inputImage = Input((nnx, nny, nnz))
        # level 1
        conv1 = self.__Conv2DBlock__(inputTensor = inputImage, filters = 64, kernelSize = 3, batchNorm = batchNorm)
        pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
        pool1 = Dropout(dropout * 0.5)(pool1)
        # level 2
        conv2 = self.__Conv2DBlock__(inputTensor = pool1, filters = 128, kernelSize = 3, batchNorm = batchNorm)
        pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
        pool2 = Dropout(dropout)(pool2)
        # level 3
        conv3 = self.__Conv2DBlock__(inputTensor = pool2, filters = 256, kernelSize = 3, batchNorm = batchNorm)
        pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
        pool3 = Dropout(dropout)(pool3)
        # level 4
        conv4 = self.__Conv2DBlock__(inputTensor = pool3, filters = 512, kernelSize = 3, batchNorm = batchNorm)
        pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
        pool4 = Dropout(dropout)(pool4)
        # feature level -- last level
        conv5 = self.__Conv2DBlock__(inputTensor = pool4, filters = 1024, kernelSize = 3, batchNorm = batchNorm)
        
        # Decoder
        # level 4
        up1 = Conv2DTranspose(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = "same")(conv5)
        up1 = concatenate([up1, conv4])
        up1 = Dropout(dropout)(up1)
        conv6 = self.__Conv2DBlock__(inputTensor = up1, filters = 512, kernelSize = 3, batchNorm = batchNorm)
        # level 3
        up2 = Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = "same")(conv6)
        up2 = concatenate([up2, conv3])
        up2 = Dropout(dropout)(up2)
        conv7 = self.__Conv2DBlock__(inputTensor = up2, filters = 256, kernelSize = 3, batchNorm = batchNorm)
        # level 2
        up3 = Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = "same")(conv7)
        up3 = concatenate([up3, conv2])
        up3 = Dropout(dropout)(up3)
        conv8 = self.__Conv2DBlock__(inputTensor = up3, filters = 128, kernelSize = 3, batchNorm = batchNorm)
        # level 1
        up4 = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = "same")(conv8)
        up4 = concatenate([up4, conv1])
        up4 = Dropout(dropout)(up4)
        conv9 = self.__Conv2DBlock__(inputTensor = up4, filters = 64, kernelSize = 3, batchNorm = batchNorm)

        # output
        outputs = Conv2D(filters = n_out, kernel_size = (1, 1), activation = 'sigmoid')(conv9)
        
        # Make model
        model = Model(inputs = inputImage, outputs = outputs)

        return model


    # NOTE: Exposed API for usage
    def SetHyperParameters(self, learning_rate = 0.001, loss = "binary_crossentropy", metrics = ["accuracy"], batchSize = 5):
        '''
        Function to set the hyperparameters
        
        Keyword Arguments:
            learning_rate {float} -- rate at which training occurs (default: {0.001})
            loss {str} -- loss function (default: {"binary_crossentropy"})
            metrics {list} -- metrics to eval (default: {["accuracy"]})
            batchSize {int} -- size of the batch (default: {5})
        '''
        self.learningRate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.batchSize = batchSize
        print("[INFO] Hyperparams set")
        return True
    
    def SetupCallbacks(self, checkpoint_dir = "./checkpoint", verbose = 1):
        '''
        Function to setup callbacks
        
        Keyword Arguments:
            checkpoint_dir {str} -- [description] (default: {"./checkpoint"})
            verbose {int} -- [description] (default: {1})
        '''
        try:
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError
            elif os.listdir(checkpoint_dir + "/") != []:
                raise FileExistsError
            else:
                True
        except FileNotFoundError:
            print("[WARN] Warning: Folder ./checkpoint/ don't exist hence making it")
            log.logger.warning("[WARN] Warning: Folder ./checkpoint/ don't exist hence making it")
            os.mkdir(checkpoint_dir)
        finally:
            self.callbacks = [
                EarlyStopping(patience = 10, verbose = verbose),
                ReduceLROnPlateau(factor = 0.1, patience = 3, min_lr = 0.000001, verbose = verbose),
                ModelCheckpoint(filepath = checkpoint_dir + "/checkpoint_model.h5", verbose = verbose, save_best_only = True, save_weights_only = True)
            ]
            print("[INFO] Callbacks set")
            return True
    
    def CompileModel(self, optimizer = "Adam") -> bool:
        '''
        Function to compile the built model
        
        Keyword Arguments:
            optimizer {str} -- the optimizer for the compilation (default: {"Adam"})
        
        Returns:
            bool -- Status check
        '''
        try:
            if not self.model:
                raise ModelFrameError
            if optimizer == "Adam":
                self.model.compile(
                    optimizer = Adam(lr = self.learningRate),
                    loss = self.loss,
                    metrics = self.metrics
                )
            else:
                self.model.compile(
                    optimizer = optimizer,
                    loss = self.loss,
                    metrics = self.metrics
                )
        except ModelFrameError:
            print("[WARN]: Build Model first!")
            log.logger.warning("[WARN]: Build Model first!")
            return False
        
        except Exception as err:
            print("[FATAL]: Compilation process failed with unknown exception: ", err)
            traceback.print_exc()
            log.logger.critical("[FATAL]: Compilation process failed with unknown exception: " + err)
            return False

        finally:
            return True
        
    def Train(self, XTrain, yTrain, XDev = None, yDev = None, n_epoch = 120):
        '''
        Function to train the data
        
        Arguments:
            XTrain {numpy array} -- input images for train 240 x 240 x 4
            yTrain {numpy array} -- label output of the image for train 240 x 240 x 1
        
        Keyword Arguments:
            XDev {[type]} -- input images for the validation 240 x 240 x 4 (default: {None})
            yDev {[type]} -- label output of the image for validation 240 x 240 x 1 (default: {None})
            n_epoch {int} -- the total number of the epoch (default: {120})
        '''
        # Set the data
        trainGen = DataGenerator(XSet = XTrain, ySet = yTrain, batch_size = self.batchSize)
        if XDev is not None and yDev is not None:
            devGen = DataGenerator(XSet = XDev, ySet = yDev, batch_size = self.batchSize)
        print("[INFO] Training started")
        log.logger.info("[INFO] Training started")

        # Train the model
        results = self.model.fit_generator(
            trainGen,
            steps_per_epoch = XTrain.shape[0] // self.batchSize,
            epochs = n_epoch,
            use_multiprocessing = True,
            shuffle = True,
            validation_data = devGen,
            validation_steps = XDev.shape[0] // self.batchSize,
            workers = 0
        )
        log.logger.info("[INFO] Train completed")
        # Plot and viz
        self.Plot(results = results)
           
    def Evaluate(self, XTest, yTest):
        '''
        Function to evaluate the model with test data
        
        Arguments:
            XTest {numpy array} -- X test data for testing
            yTest {numpy array} -- y label test data for testing
        '''

        testGen = DataGenerator(XTest, yTest, self.batchSize)
        self.model.evaluate_generator(
            testGen,
            steps = XTest.shape[0] // self.batchSize,
            verbose = 1,
            use_multiprocessing = True,
            workers = 0)
    
    def Predict(self, XTest, yTest, folderPath = "./test"):
        '''
        Function to predict the output and save them in the folder path given
        
        Arguments:
            XTest {numpy array} -- the X numpy array of test data
            yTest {numpy array} -- the y label numpy array of test data
        
        Keyword Arguments:
            folderPath {str} -- [description] (default: {"./test"})
        
        Returns:
            bool -- status
        '''

        pred = self.model.predict(XTest)
        for ind, (x, y) in enumerate(zip(XTest, yTest)):
            VisualizeImageWithPrediction(x, y, pred, path = folderPath + "/test_pred_{}.png".format(ind), )
        return True

    def Plot(self, results):
        '''
        Function to plot the given results from keras train
        
        Arguments:
            results {History} -- keras History class object
        '''

        plt.figure(figsize = (8, 8))
        plt.title("Learning Curve")
        plt.plot(results.history["loss"], label = "loss")
        plt.plot(results.history["val_loss"], label = "val_loss")
        plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker = "x", color = "r", label = "best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()
        plt.show()
        plt.savefig("./plots/results.png")
        plt.clf()
        
    def SaveModel(self, path = "./savedModels/", modelName = "UNet_Standard"):
        '''
        Function to save the model in the given directory
        
        Keyword Arguments:
            path {str} -- path to save the model (default: {"./savedModels/"})
            modelName {str} -- the name of the model (default: {"UNet_Standard"})
        '''
        log.logger.info("[INFO] Saving model as JSON and h5")
        SaveModelJSON(self.model, modelName, path)
    
    def LoadModelWeights(self, path = "./checkpoint/checkpoint_model.h5"):
        '''
        Function to load the weights alone given in .h5 file in the path

        Keyword Arguments:
            path {str} -- path for the weights (default: {"./checkpoint/checkpoint_model.h5"})
        '''
        self.model.load_weights(path)
        print("[INFO] Weight loaded")
        log.logger.info("[INFO] Weight loaded")
        return True
    
    def LoadModel(self, path = "./savedModels/", modelName = "UNet_Standard"):
        '''
        Function to load the model
        
        Keyword Arguments:
            path {str} -- the path for the saved model (default: {"./savedModels/"})
            modelName {str} -- the name of the model (default: {"UNet_Standard"})
        '''

        self.model = LoadModelJSON(modelName, path)
        print("[INFO] Model Loaded")
        log.logger.info("[INFO] Model loaded")
        return True
