from keras.models import *
from keras.layers import *
import numpy as np
import tensorflow as tf

def AttentionBlock(x, shortcut, num_filters):
    '''
    function that defines the attention gated block for our U-Net Model

    Arguments:
        x {tensor} -- Input from previous decoding upscale layer
        shortcut {tensor} -- Corresponding input from the same level encoder layer
        num_filters {int} -- total filters used for that layer in decoder

    Returns:
        {tensor} -- the output of the attention gate to be concatenated with that layer in decoder
    '''
    g1 = Conv2D(num_filters, kernel_size = 1, kernel_initializer = "he_normal", padding = "same")(shortcut)
    x1 = Conv2D(num_filters, kernel_size = 1, kernel_initializer = "he_normal", padding = "same")(x)
    
    g1_x1 = Add()([g1, x1])
    psi = Activation("relu")(g1_x1)
    psi = Conv2D(1, kernel_size = 1, padding = "same")(psi)
    psi = Activation("sigmoid")(psi)
    x = Multiply()([x, psi])
    return x

def UNet(shape, reuse = False, n_out = 1):
    
    nnx = int(shape[1])
    nny = int(shape[2])
    nnz = int(shape[3])
    print(" * Input: size of image: %d %d %d" % (nnx, nny, nnz))
    with tf.variable_scope("u_net", reuse = reuse):
        
        # Encoder
        inputs = Input((nnx, nny, nnz))
        
        conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(inputs)
        conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv1)
        pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool1)
        conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv2)
        pool2 = MaxPool2D(pool_size = (2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool2)
        conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv3)
        pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool3)
        conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv4)
        conv4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
        
        conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool4)
        conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv5)
        conv5 = Dropout(0.5)(conv5)
        
        # Decoder ;_;
        up6 = Deconv2D(512, 3, strides = (2, 2), padding = "same")(conv5)
        merge6 = concatenate([conv4, up6], axis = -1)
        conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge6)
        conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv6)
        
        up7 = Deconv2D(256, 3, strides = (2, 2), padding = "same")(conv6)
        merge7 = concatenate([conv3, up7], axis = -1)
        conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge7)
        conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv7)
        
        up8 = Deconv2D(128, 3, strides = (2, 2), padding = "same")(conv7)
        merge8 = concatenate([conv2, up8], axis = -1)
        conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge8)
        conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv8)
        
        up9 = Deconv2D(64, 3, strides = (2, 2), padding = "same")(conv8)
        merge9 = concatenate([conv1, up9], axis = -1)
        conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge9)
        conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv9)
        conv9 = Conv2D(2, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv9)

        conv10 = Conv2D(n_out, 1, activation = "sigmoid", padding = "same", kernel_initializer = "he_normal")(conv9)
        
        model = Model(inputs = inputs, outputs = conv10)
        
        return model
    
def UNetWithAttention(shape, reuse = False, n_out = 1):
    
    nnx = int(shape[1])
    nny = int(shape[2])
    nnz = int(shape[3])
    print(" * Input: size of image: %d %d %d" % (nnx, nny, nnz))
    with tf.variable_scope("u_net", reuse = reuse):
        
        # Encoder
        inputs = Input((nnx, nny, nnz))
        
        conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(inputs)
        conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv1)
        pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool1)
        conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv2)
        pool2 = MaxPool2D(pool_size = (2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool2)
        conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv3)
        pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool3)
        conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv4)
        conv4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
        
        conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(pool4)
        conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv5)
        conv5 = Dropout(0.5)(conv5)
        
        # Decoder ;_;
        up6 = Deconv2D(512, 3, strides = (2, 2), padding = "same")(conv5)
        attn6 = AttentionBlock(up6, conv4, 512)
        merge6 = concatenate([attn6, up6], axis = -1)
        conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge6)
        conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv6)
        
        up7 = Deconv2D(256, 3, strides = (2, 2), padding = "same")(conv6)
        attn7 = AttentionBlock(up7, conv3, 256)
        merge7 = concatenate([attn7, up7], axis = -1)
        conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge7)
        conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv7)
        
        up8 = Deconv2D(128, 3, strides = (2, 2), padding = "same")(conv7)
        attn8 = AttentionBlock(up8, conv2, 128)
        merge8 = concatenate([attn8, up8], axis = -1)
        conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge8)
        conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv8)
        
        up9 = Deconv2D(64, 3, strides = (2, 2), padding = "same")(conv8)
        attn9 = AttentionBlock(up9, conv1, 64)
        merge9 = concatenate([attn9, up9], axis = -1)
        conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(merge9)
        conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv9)
        conv9 = Conv2D(2, 3, activation = "relu", padding = "same", kernel_initializer = "he_normal")(conv9)

        conv10 = Conv2D(n_out, 1, activation = "sigmoid", padding = "same", kernel_initializer = "he_normal")(conv9)
        
        model = Model(inputs = inputs, outputs = conv10)
        
        return model