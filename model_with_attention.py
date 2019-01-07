import tensorflow as tf
import tensorlayer as tl
import numpy as np


def AttentionBlock(x, shortcut, num_filters, isTrain, gammaInit, bInit, n = None):
    '''
    function that defines the attention gated block for our U-Net Model
    
    Arguments:
        x {tensor} -- Input from previous decoding upscale layer
        shortcut {tensor} -- Corresponding input from the same level encoder layer
        num_filters {int} -- total filters used for that layer in decoder
        
    Returns:
        {tensor} -- the output of the attention gate to be concatenated with that layer in decoder
    '''
    g1 = tl.layers.Conv2d(shortcut, num_filters, (1, 1), name = "g1conv1" + n)
    g1 = tl.layers.BatchNormLayer(g1, is_train = isTrain, gamma_init = gammaInit, beta_init = bInit, name = "g1bn1" + n)
    x1 = tl.layers.Conv2d(x, num_filters, (1, 1), name = "x1conv1" + n)
    x1 = tl.layers.BatchNormLayer(x1, is_train = isTrain, gamma_init = gammaInit, beta_init = bInit, name = "x1bn1" + n)
    g1_x1 = tf.add(g1, x1)
    psi = tf.nn.relu(g1_x1)
    psi = tl.layers.Conv2d(psi, 1, (1, 1), name = "psiconv1" + n)
    psi = tl.layers.BatchNormLayer(psi, is_train = isTrain, gamma_init = gammaInit, beta_init = bInit, name = "psibn" + n)
    psi = tf.nn.sigmoid(psi)
    x = tf.multiply(x, psi)
    return x

def UNetWithAttention(x, isTrain = False, reuse = False, batchSize = None, pad = "SAME", nOut = 1):
    '''
    Function that defines the UNet model with attention
    '''
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    nz = int(x.shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
    
    wInit = tf.truncated_normal_initializer(stddev = 0.01)
    bInit = tf.constant_initializer(value = 0.0)
    gammaInit = tf.random_normal_initializer(mean = 1., stddev = 0.02)
    
    with tf.variable_scope("u_net_attention", reuse = reuse):
        tl.layers.set_name_reuse(reuse)
        # Define input layer
        inputs = tl.layers.InputLayer(x, name = "inputs")
        
        # Encoder 
        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')
        
        # Decoder ;_;
        up4 = tl.layers.DeConv2d(prev_layer = conv5, n_filter = 512, filter_size = (3, 3), strides = (2, 2), name = "deconv4")
        up4 = AttentionBlock(x = up4, shortcut = conv4, num_filters = 512, isTrain=isTrain, gammaInit=gammaInit, bInit=bInit, n = "atn4")
        up4 = tl.layers.ConcatLayer([up4, conv4], 3, name = "concat4")
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = tl.layers.DeConv2d(prev_layer = conv4, n_filter = 256, filter_size = (3, 3), strides = (2, 2), name = "deconv3")
        up3 = AttentionBlock(x = up3, shortcut = conv3, num_filters = 256, isTrain=isTrain, gammaInit=gammaInit, bInit=bInit, n = "atn3")
        up3 = tl.layers.ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = tl.layers.DeConv2d(prev_layer = conv3, n_filter = 128, filter_size = (3, 3),  strides = (2, 2), name = "deconv2")
        up2 = AttentionBlock(x = up2, shortcut = conv2, num_filters = 128, isTrain=isTrain, gammaInit=gammaInit, bInit=bInit, n = "atn2")
        up2 = tl.layers.ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = tl.layers.DeConv2d(prev_layer = conv2, n_filter = 64, filter_size = (3, 3), strides = (2, 2), name = "deconv1")
        up1 = AttentionBlock(x = up1, shortcut = conv1, num_filters = 64, isTrain=isTrain, gammaInit=gammaInit, bInit=bInit, n = "atn1")
        up1 = tl.layers.ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = tl.layers.Conv2d(conv1, nOut, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    
    return conv1