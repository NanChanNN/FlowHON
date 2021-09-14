from nn.layer import myLayerLinear_V2
from nn.layer import myLayerSoftmax
from nn.layer import myLayerDivision_V2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utility import *
from nn.dataSetLoader import HoNLoader
from nn.dataSetLoader import HoNCluLoader

class trainer():
    def __init__(self, flowName, manipulation, graphType, data_loader, training_model = None):
        self.flowName = flowName
        self.manipulation = manipulation
        self.graphType = graphType

        #self.Docu = Docu
        #self.training_file = training_file
        #self.validation_file = validation_file
        self.HoN = data_loader

        #self.configure_parameter(100, 10, self.HoN.clusterSize, 1e-2, "Adamax", "KLD")
        self.configure_parameter(100, 10, self.HoN.clusterSize, 1e-2, "Adamax", "MSE")
        self.factor = 1000 / self.sampleSize
        #self.set_network()
        self.set_data_set()
        if training_model is None:
            self.set_training_model()
        else:
            self.testLayer = training_model
        self.trainingRecord = {}

    def configure_parameter(self, epoches, batchSize, sampleSize, lr, optimizer, loss):
        self.epoches = epoches
        self.batchSize = batchSize
        self.sampleSize = sampleSize
        self.learningRate = lr
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10,
            decay_rate=0.9)

        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        elif optimizer == "Adamax":
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=self.lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

        if loss == "KLD":
            self.loss_fn = tf.keras.losses.KLDivergence()
        else:
            self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.train_acc_metric = tf.keras.metrics.KLDivergence()
        self.val_acc_metric = tf.keras.metrics.KLDivergence()

    def set_network(self):
        if self.graphType == "Clustering":
            self.HoN = HoNCluLoader(self.flowName, '', self.Docu, self.training_file, self.validation_file)
        elif self.graphType == "Ref":
            self.HoN = HoNCluLoader(self.flowName, 'Ref', self.Docu, self.training_file, self.validation_file)
        else:
            self.HoN = HoNLoader(self.flowName, self.graphType, self.Docu, self.training_file, self.validation_file)
    '''
    def set_data_set(self):
        x_train, y_train, x_valid, y_valid = self.HoN.splitTrainAndValid(splitRate=0.0)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = train_dataset.batch(self.batchSize)

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        self.validation_dataset = validation_dataset.batch(self.batchSize)
    '''
    def set_data_set(self):
        x_train, y_train, y_train_2, x_valid, y_valid, y_valid_2 = self.HoN.splitTrainAndValid_V2(splitRate=0.0)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, y_train_2))
        self.train_dataset = train_dataset.batch(self.batchSize)

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid, y_valid_2))
        self.validation_dataset = validation_dataset.batch(self.batchSize)

    def set_training_model(self):
        if (self.manipulation == "Linear"):
            self.testLayer = myLayerLinear_V2(self.HoN.vertexNumber, self.HoN.regionNumber, self.HoN.getTransitionProb(),
                                      self.HoN.getMapNode2Node(), self.HoN.getMask())
        elif (self.manipulation == "Softmax"):
            self.testLayer = myLayerSoftmax(self.HoN.vertexNumber, self.HoN.regionNumber, self.HoN.getTransitionProb(),
                                       self.HoN.getMapNode2Node(), self.HoN.getMask())
        elif (self.manipulation == "Division"):
            self.testLayer = myLayerDivision_V2(self.HoN.vertexNumber, self.HoN.regionNumber, self.HoN.getTransitionProb(),
                                        self.HoN.getMapNode2Node(), self.HoN.getMask())

    def get_transition_mat(self):
        return self.testLayer.trainable_weights[0].numpy()

    '''
    def run(self, epoches=None):
        if epoches is None:
            epoches = self.epoches

        totalTrainingLoss = []
        variance = []

        for epoch in range(epoches):
            tmp_loss = []
            prevMatrix = self.testLayer.trainable_weights[0].numpy()

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.testLayer(x_batch_train)

                    loss_value = self.loss_fn(y_batch_train / self.sampleSize, logits / self.sampleSize)
                    #loss_value = tf.keras.losses.MeanSquaredError()(y_batch_train, logits) #MSE
                    loss_value += sum(self.testLayer.losses)

                    #totalTrainingLoss.append(loss_value)

                # prevMatrix = testLayer.trainable_weights[0].numpy()
                grads = tape.gradient(loss_value, self.testLayer.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.testLayer.trainable_weights))
                tmp_loss.append(loss_value.numpy())

            totalTrainingLoss.append(np.mean(tmp_loss))
            variance.append(np.linalg.norm(self.testLayer.trainable_weights[0].numpy() - prevMatrix, ord=2))

        self.trainingRecord['variance'] = variance
        self.trainingRecord['training loss record'] = totalTrainingLoss
    '''

    def run(self, epoches=None, s_n_map = None):
        if epoches is None:
            epoches = self.epoches

        totalTrainingLoss = []
        variance = []

        # projection gradient descent
        tmp_matrix = self.get_transition_mat()
        tmp_matrix = np.clip(tmp_matrix, a_min=0.0, a_max=10.0)
        self.testLayer.trainable_weights[0].assign(tmp_matrix)

        for epoch in range(epoches):
            tmp_loss = []
            prevMatrix = self.testLayer.trainable_weights[0].numpy()

            for step, (x_batch_train, y_batch_train, y_2_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits, logits_2 = self.testLayer(x_batch_train)

                    if s_n_map is not None:
                        y_2_batch_train = tf.matmul(y_2_batch_train, s_n_map)

                    loss_value_1 = self.loss_fn(y_batch_train * self.factor, logits * self.factor)
                    loss_value_2 = self.loss_fn(y_2_batch_train * self.factor, logits_2 * self.factor)
                    loss_value = loss_value_2 + loss_value_1
                    #loss_value = tf.keras.losses.MeanSquaredError()(y_batch_train, logits) #MSE
                    loss_value += sum(self.testLayer.losses)

                    #totalTrainingLoss.append(loss_value)

                # prevMatrix = testLayer.trainable_weights[0].numpy()
                grads = tape.gradient(loss_value, self.testLayer.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.testLayer.trainable_weights))

                tmp_loss.append(loss_value.numpy())

            totalTrainingLoss.append(np.mean(tmp_loss))
            variance.append(np.linalg.norm(self.testLayer.trainable_weights[0].numpy() - prevMatrix, ord=2))

        self.trainingRecord['variance'] = variance
        self.trainingRecord['training loss record'] = totalTrainingLoss