import tensorflow as tf
from tensorflow import keras
import numpy as np
import nn.dataSetLoader
tf.keras.backend.set_floatx('float32')


class myLayerLinear(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, wM, tM, mM, isRandom=False):
        super(myLayerLinear, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5)
        if isRandom:
            self.w = tf.Variable(initial_value=self.initializer(shape=wM.shape), trainable=True, dtype=tf.float32)
        else:
            self.w = tf.Variable(initial_value=wM, trainable=True, dtype=tf.float32)
        self.relu = tf.keras.layers.ReLU()
        #self.w = tf.Variable(initial_value=wM, trainable=True, dtype=tf.float32)
        self.tran = tf.Variable(initial_value=tM, trainable=False, dtype=tf.float32)
        self.mask = tf.Variable(initial_value=mM, trainable=False, dtype=tf.float32)

        self.alpha1=0.1 * 10
        self.alpha2=0.1 * 10
        self.alpha3=0.1 * 10

    def call(self, inputs):
        #forward pass
        #reluW = self.relu(self.w)
        maskW = self.w * self.mask
        resultPro = tf.matmul(inputs, maskW)

        #penalty term
        self.my_regularizer(maskW)

        return tf.matmul(resultPro, self.tran)

    def my_regularizer(self, W):
        cumulate = tf.keras.backend.sum(W, axis=1)
        constant = tf.keras.backend.ones(shape=cumulate.shape, dtype=tf.float32)
        subs = tf.math.subtract(cumulate, constant)
        
        loss = self.alpha1 * 1/2 * tf.reduce_sum(tf.square(subs))
        loss += self.alpha2 * tf.reduce_sum(self.relu(tf.math.negative(W)))
        loss += self.alpha3 * tf.reduce_sum(self.relu(W-1))

        self.add_loss(loss)


class myLayerSoftmax(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, wM, tM, mM, isRandom=False):
        super(myLayerSoftmax, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal()
        if isRandom:
            self.w = tf.Variable(initial_value=self.initializer(shape=wM.shape), trainable=True)
        else:
            self.w = tf.Variable(initial_value=wM, trainable=True)
        self.tran = tf.Variable(initial_value=tM, trainable=False)
        self.mask = tf.Variable(initial_value=mM, trainable=False)

        self.penalty = tf.keras.regularizers.l2(0.01)

    def call(self, inputs):

        self.add_loss(self.penalty(self.w))

        #modify transition probability matrix
        expW = tf.exp(self.w) * self.mask
        proMatrix = expW / tf.reduce_sum(expW, 1, keepdims=True)

        #forward pass
        resultPro = tf.matmul(inputs, proMatrix)

        return tf.matmul(resultPro, self.tran)


class myLayerDivision(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, wM, tM, mM, isRandom=False):
        super(myLayerDivision, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal()
        if isRandom:
            self.w = tf.Variable(initial_value=self.initializer(shape=wM.shape), trainable=True)
        else:
            self.w = tf.Variable(initial_value=wM, trainable=True)
            
        self.tran = tf.Variable(initial_value=tM, trainable=False)
        self.mask = tf.Variable(initial_value=mM, trainable=False)

        self.penalty = tf.keras.regularizers.l2(0.01)

    def call(self, inputs):

        #self.add_loss(self.penalty(self.w))

        #modify transition probability matrix
        maskW = self.w * self.mask
        absoluteW = tf.abs(maskW)

        proMatrix = absoluteW / tf.reduce_sum(absoluteW, 1, keepdims=True)

        #forward pass
        resultPro = tf.matmul(inputs, proMatrix)

        return tf.matmul(resultPro, self.tran)


class myLayerDivision_V2(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, wM, tM, mM, isRandom=False):
        super(myLayerDivision_V2, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal()
        if isRandom:
            self.w = tf.Variable(initial_value=self.initializer(shape=wM.shape), trainable=True)
        else:
            self.w = tf.Variable(initial_value=wM, trainable=True)

        self.tran = tf.Variable(initial_value=tM, trainable=False)
        self.mask = tf.Variable(initial_value=mM, trainable=False)

        self.penalty = tf.keras.regularizers.l2(0.01)

    def call(self, inputs):

        # self.add_loss(self.penalty(self.w))

        # modify transition probability matrix
        maskW = self.w * self.mask
        absoluteW = tf.abs(maskW)

        proMatrix = absoluteW / tf.reduce_sum(absoluteW, 1, keepdims=True)

        # forward pass
        resultPro = tf.matmul(inputs, proMatrix)

        return tf.matmul(resultPro, self.tran), resultPro

    def regularize_weight(self):
        pass

    def get_trainsition_matrix(self):
        maskM = self.w.numpy() * self.mask.numpy()
        absoluteW = np.abs(maskM)
        proMatrix = absoluteW / np.sum(absoluteW, 1, keepdims=True)

        return proMatrix


class myLayerLinear_V2(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, wM, tM, mM, isRandom=False):
        super(myLayerLinear_V2, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5)
        if isRandom:
            self.w = tf.Variable(initial_value=self.initializer(shape=wM.shape), trainable=True, dtype=tf.float32)
        else:
            self.w = tf.Variable(initial_value=wM, trainable=True, dtype=tf.float32)
        self.relu = tf.keras.layers.ReLU()
        # self.w = tf.Variable(initial_value=wM, trainable=True, dtype=tf.float32)
        self.tran = tf.Variable(initial_value=tM, trainable=False, dtype=tf.float32)
        self.mask = tf.Variable(initial_value=mM, trainable=False, dtype=tf.float32)

        self.alpha1 = 0.1 * 5
        self.alpha2 = 0.1 * 5
        self.alpha3 = 0.1 * 5

    def call(self, inputs):
        # forward pass
        # reluW = self.relu(self.w)
        maskW = self.w * self.mask
        resultPro = tf.matmul(inputs, maskW)

        # penalty term
        self.my_regularizer(maskW)

        return tf.matmul(resultPro, self.tran), resultPro

    def my_regularizer(self, W):
        cumulate = tf.keras.backend.sum(W, axis=1)
        constant = tf.keras.backend.ones(shape=cumulate.shape, dtype=tf.float32)
        subs = tf.math.subtract(cumulate, constant)

        loss = self.alpha1 * 1 / 2 * tf.reduce_sum(tf.square(subs))
        loss += self.alpha2 * tf.reduce_sum(self.relu(tf.math.negative(W)))
        loss += self.alpha3 * tf.reduce_sum(self.relu(W - 1))

        self.add_loss(loss)

    def regularize_weight(self):
        #pass
        proMatrix = self.get_trainsition_matrix()
        self.w.assign(proMatrix)

    def get_trainsition_matrix(self):
        maskM = self.w.numpy() * self.mask.numpy()
        maskM = np.clip(maskM, a_min=0, a_max=None)
        proMatrix = maskM / np.sum(maskM, 1, keepdims=True)

        return proMatrix

