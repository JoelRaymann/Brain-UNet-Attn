import tensorflow as tf
import tensorlayer as tl
import os, time, traceback
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
import cv2 as cv

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

def VisualizeImageWithPrediction(X, y, yPred, path = "test_pred.png", intensity = 1.5):
    '''
    Function to store one slice with prediction
    This function combines X with y(ground truth) and yPred(prediction) and
    saves them in path
    
    Arguments:
        X {tensor} -- The X input data with dims (height, width, no_of_X_input(4))
        y {tensor} -- The ground truth value with dims (height, width)
        yPred {tensor} -- The ground truth value with dims (height, width)
        path {str} -- the path to save the image
        intensity {float} -- the intensity value to set

    Returns:
        None
    '''
    if y.ndim == 2:
        # Then add a dimension to make it have channels
        y = y[:, :, np.newaxis] # (height, width, channel = 1)
    
    if yPred.ndim == 2:
        yPred = yPred[:, :, np.newaxis]
    
    assert X.ndim == 3 # Check dimensions of X
    
    X = (np.copy(X) * 1.5).astype(np.uint8)
    y = ((np.copy(y) > 0) * 255.).astype(np.uint8)
    yPred = ((np.copy(yPred) > 0) * 255.).astype(np.uint8)
    img = np.concatenate((X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis], y, yPred), axis = 1)
    img = cv.applyColorMap(img, cv.COLORMAP_HOT) 
    cv.imwrite(path, img)
    
def VisualizeImage(X, y, path = "test.png", intensity = 1.5):
    '''
    Function to store one image slice in given path.
    This function combines all images of X with y and stores it in
    path as one image
    
    Arguments:
        X {tensor} -- input data X
        y {tensor} -- ground truth value y
        path {str} -- path to save the file {test.png}
        intensity {float} -- the value to increase the intensity
    Returns:
        None
    '''
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    
    assert X.ndim == 3 # Make sure the X consist of dim - (height, width, no_of_X_data(4))
    
    X = (np.copy(X) * 1.5).astype(np.uint8)
    y = ((y > 0) * 255.).astype(np.uint8)
    img = np.concatenate((X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis], y), axis = 1)
    img = cv.applyColorMap(img, cv.COLORMAP_HOT) 
    cv.imwrite(path, img)   
