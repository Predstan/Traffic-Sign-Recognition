

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import cv2
import numpy as np


def batch_norm(x, out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps

    """

    

    with tf.compat.v1.variable_scope('bn'):
        beta = tf.compat.v1.Variable(tf.constant(0.0, shape=[out]),
                                     name='beta', trainable=True)

        gamma = tf.compat.v1.Variable(tf.constant(1.0, shape=[out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



def convolution2D(X, kernel, filter, name, strides=1, padding="VALID", activation=None, mean=0, sigma=0.1):
    """
    Implement a Convolutional Step
    Arguments:
        X -- Input Tensor
        kernel -- Kernel Size of type integer
        filter -- Size of filter of type integer

    """

    shape = X.get_shape()
    print(shape)

    #print(shape)
    w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = (kernel, kernel, shape[3], filter), mean=mean, stddev=sigma))
    b = tf.compat.v1.Variable(tf.zeros(filter))
    X = tf.nn.conv2d(X, w, strides = [1, strides, strides, 1], padding=padding, name= name) + b
    if activation is not None:
        return tf.nn.relu(X)

    return X
    



def MyNet(input_shape= (32, 32, 3), classes=43, mean=0, sigma=0.1, training=True, dropout_rate=0.5):

    X_input = tf.compat.v1.placeholder(tf.float32, shape=[None] + list(input_shape))

    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')

    train_tensor = tf.compat.v1.placeholder(tf.bool, (None))

    Y = tf.compat.v1.placeholder(tf.int32, (None))

    one_hot_y  = tf.one_hot(Y, classes)


    X = convolution2D(X_input, filter = 64, kernel=3, strides= 1, name="conv_1")
    X = tf.nn.relu(X)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    X = convolution2D(X, filter = 128, kernel=3, strides= 1, name="conv_2")
    if training:
        X = batch_norm(X, 128, train_tensor)

    X = tf.nn.relu(X)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')


    X = convolution2D(X, filter = 256, kernel=3, strides= 1, name="conv_3")

    if training:
        X = batch_norm(X, 256, train_tensor)

    X = tf.nn.relu(X)
    X = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    if training:
        X = tf.nn.dropout(X, rate =dropout_rate)

    shape = X.get_shape()

    conv_output_width = shape[2]
    conv_output_height = shape[1]

    conv_element_count = int(
        conv_output_width * conv_output_height * shape[3])

    X = tf.reshape(X,[-1, conv_element_count])


    fc_W  = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape=(conv_element_count, 256), mean = mean, stddev = sigma))
    fc_b  = tf.compat.v1.Variable(tf.zeros(256))

    X = tf.matmul(X, fc_W) + fc_b


    fc_W  = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape=(256, classes), mean = mean, stddev = sigma))
    fc_b  = tf.compat.v1.Variable(tf.zeros(classes))

    logits = tf.matmul(X, fc_W) + fc_b

    return  X_input, Y, one_hot_y, logits, train_tensor, dropout_rate



def preprocess_input(image):
    
    shape = image.shape
    if shape[-2] == 32 and shape[-3] == 32:
        im = image/255.0
        return im

    if len(shape) == 4:
        all_image = []
        for im in image:
            all_image.append(np.expand_dims(cv2.resize(im, (32, 32) )/255, 0))
        im = np.concatenate(all_image)
    else:
        im = cv2.resize(image, (32, 32) )/255.
    return im


