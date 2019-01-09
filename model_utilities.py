# Code consisting of all the necessary preprocessing and post processing utilities
import tensorflow as tf
import tensorlayer as tl
import os, time, traceback
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json

def DistortImages(data):
    '''
    Function to augment the data with Tensorlayer
    
    Arguments:
        data {tensor} -- The input to do data augmentation
        
    Returns:
        x1, x2, x3, x4, y {tensors} -- The augmented tensors of the data
    '''
    
    # Fetch data
    x1, x2, x3, x4, y = data
    
    # Apply Augmentation
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis = 1, is_random = True) # Axis = 1 => left right flip
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y], alpha = 720, sigma = 24, is_random = True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg = 20, is_random = True, fill_mode = "constant")
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg = 0.10, hrg = 0.10, is_random = True, fill_mode = "constant")
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05, is_random = True, fill_mode = "constant")
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y], zoom_range = [0.9, 1.1], is_random = True, fill_mode = "constant")
    return x1, x2, x3, x4, y

def VisualizeImageWithPrediction(X, y, yPred, path):
    '''
    Function to store one slice with prediction
    This function combines X with y(ground truth) and yPred(prediction) and
    saves them in path
    
    Arguments:
        X {tensor} -- The X input data with dims (height, width, no_of_X_input(4))
        y {tensor} -- The ground truth value with dims (height, width)
        yPred {tensor} -- The ground truth value with dims (height, width)
        
    Returns:
        None
    '''
    if y.ndim == 2:
        # Then add a dimension to make it have channels
        y = y[:, :, np.newaxis] # (height, width, channel = 1)
    
    if yPred.ndim == 2:
        yPred = yPred[:, :, np.newaxis]
    
    assert X.ndim == 3 # Check dimensions of X
    
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis], X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis], y, yPred]),
                                 size = (1, 6),
                                 image_path = path)

def VisualizeImage(X, y, path):
    '''
    Function to store one image slice in given path.
    This function combines all images of X with y and stores it in
    path as one image
    
    Arguments:
        X {tensor} -- input data X
        y {tensor} -- ground truth value y
        path {str} -- path to save the file
    
    Returns:
        None
    '''
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    
    assert X.ndim == 3 # Make sure the X consist of dim - (height, width, no_of_X_data(4))
    
    tl.vis.save_images(np.asarray([X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis], y]),
                       size = (1, 5), 
                       image_path = path)

def InitializeFolders(folderList: list) -> bool:
    '''
    Function to initialize all the required folders for the model to
    run
    
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
            print("[+] Exception occured while making folders: ", err)
            traceback.print_exc()
            return False
    print("[+] Folders initialized")
    return True

def dice_coef(y_true, y_pred, smooth = 1):
    '''
    Function that defines the dice_coef
    Dice = (2*|X & Y|)/ (|X|+ |Y|) = 2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    '''
    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    return (2. * intersection + smooth)/(K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def SaveModelJSON(model: Model, modelName: str,save_dir = "") -> bool:
    '''
    Function to save the model to a JSON format with
    weights as h5py for later loading and referencing
    
    Arguments:
        model {Model} -- Keras trained model
        modelName {str} -- name of the model
    Keyword Arguments:
        save_dir {str} -- path to save the model (default: {""})
    
    Returns:
        bool -- status check
    '''
    modelJSON = model.to_json()
    with open(save_dir + modelName + ".json", "w") as f:
        f.write(modelJSON)
    
    # Save weights
    model.save_weights(save_dir + modelName + ".h5")
    print("[+] Model saved in disk")
    return True

def LoadModelJSON(modelName:str, load_dir = "") -> Model:
    '''
    Function to load the JSON model and return it
    
    Arguments:
        modelName {str} -- name of the model to load
    
    Keyword Arguments:
        load_dir {str} -- The model's saved directory (default: {""})
    
    Returns:
        Model -- the trained keras model -- need to compile!!!
    '''
    jsonFile = open(load_dir + modelName + ".json", "r")
    loadedModelJson = jsonFile.read()
    jsonFile.close()

    model = model_from_json(loadedModelJson)
    model.load_weights(load_dir + modelName + ".h5")
    print("[+] Model loaded. Please compile it")
    return model
