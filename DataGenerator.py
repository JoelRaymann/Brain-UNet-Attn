import tensorflow as tf
import tensorlayer as tl
import numpy as np
from data_utilities import DistortImages
from keras import backend as K
from keras.utils import Sequence
from random import shuffle
import math

class DataGenerator(Sequence):

    def __init__(self, XSet, ySet, batch_size):
        self.X, self.y = XSet, ySet
        self.batch_size = batch_size
        _ , self.nw, self.nh, self.nz = self.X.shape
        
    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        images = self.X[index * self.batch_size : (index + 1) * self.batch_size, :, :, :]
        labels = self.y[index * self.batch_size : (index + 1) * self.batch_size, :, :, :]
        data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                        images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                        images[:,:,:,3, np.newaxis], labels)], fn = DistortImages)
        bImages = data[:, 0:4, :, :, :]
        bLabels = data[:, 4, :, :, :]
        bImages = bImages.transpose((0, 2, 3, 1, 4))
        bImages.shape = (self.batch_size, self.nw, self.nh, self.nz)
        return bImages, bLabels

    def on_epoch_end(self):
        
        indexList = list(range(self.X.shape[0]))
        shuffle(indexList)
        self.X = self.X[indexList, :, :, :]
        self.y = self.y[indexList, :, :, :]