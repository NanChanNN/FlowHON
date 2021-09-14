import numpy as np
import tensorflow as tf
from tensorflow import keras
from nn.utls import check_unique_axis
tf.keras.backend.set_floatx('float32')

class dual_layer(keras.layers.Layer):
    def __init__(self, mat_A, mat_B, mat_T, mat_M, train_A=False, train_B=False):
        super(dual_layer, self).__init__()

        # create trainable matrix A and B
        # A with size m * n, where A_ij denotes that node i contains status j
        # B with size m*m, where B_ij denotes the probability that node i to node j
        self.mat_A = tf.Variable(initial_value=mat_A, trainable=train_A, name='Matrix A', dtype=tf.float32)
        self.mat_B = tf.Variable(initial_value=mat_B, trainable=train_B, name='Matrix B', dtype=tf.float32)

        # create fixed matrix T and M
        # T with size n * l, where T_ij denotes that status i belongs to block j
        # M with size n * n, where M_ij denotes that status i can transform to status j
        self.mat_T = tf.Variable(initial_value=mat_T, trainable=False, name='Matrix T', dtype=tf.float32)
        self.mat_M = tf.Variable(initial_value=mat_M, trainable=False, name='Matrix M', dtype=tf.float32)

        self.entropy = tf.keras.losses.categorical_crossentropy
        self.relu = tf.keras.layers.ReLU()
        self.alpha1 = 0.1
        self.alpha2 = 0.1
        self.alpha3 = 0.1
        self.proMatrix = None
        self.train_A = train_A
        self.train_B = train_B

        self.inter_result = None

        self.get_mask()
        m_B = mat_B * self.mask
        self.mat_B = tf.Variable(initial_value=m_B, trainable=train_B, name='Matrix B', dtype=tf.float32)

    def call(self, inputs):
        """
        Calculate one iteration for inputs,
        :param inputs: input data, with size n
        :return: output data, with size l
        """
        inputs = tf.transpose(inputs)

        # Step 1
        temp_1 = tf.matmul(self.mat_A, inputs)

        # Step 2
        #tmp_B = self.mat_B * self.mask
        tmp_B = self.mat_B
        temp_2 = tf.matmul(tf.transpose(tmp_B), temp_1)

        # Step 3
        temp_Mat = tf.clip_by_value(tf.matmul(self.mat_A, self.mat_T), clip_value_min=0, clip_value_max=1)

        # Step 4
        outputs = tf.matmul(tf.transpose(temp_Mat), temp_2)

        # Set_penalty
        self.penalty_1()
        self.penalty_2(temp_Mat)
        self.my_regularizer(self.mat_A, axis=0)
        self.penalty_3(tmp_B) # self.mat_B
        self.my_regularizer(tmp_B)

        return tf.transpose(outputs)

    def inference(self, inputs, alongTime = True):
        """
        Calculate one iteration for inputs,
        :param inputs: input data, with size n
        :return: output data, with size l
        """
        if self.inter_result is None or alongTime is False:
            if inputs.dtype != 'float32':
                inputs = tf.cast(inputs, dtype=tf.float32)
            inputs = tf.transpose(inputs)

            # Step 1
            temp_1 = tf.matmul(self.mat_A, inputs)
            self.inter_result = temp_1

        temp_1 = self.inter_result

        # Step 2
        if self.proMatrix is None:
            self.regularize_mat_B()
        tmp_B = self.proMatrix * self.mask

        temp_2 = tf.matmul(tf.transpose(tmp_B), temp_1)
        self.inter_result = temp_2

        # Step 3
        temp_Mat = tf.clip_by_value(tf.matmul(self.mat_A, self.mat_T), clip_value_min=0, clip_value_max=1)

        # Step 4
        outputs = tf.matmul(tf.transpose(temp_Mat), temp_2)

        return tf.transpose(outputs)

###################### regularization ###################
    def get_mask(self):
        mask_M = tf.matmul(self.mat_A, self.mat_M)
        self.mask = tf.clip_by_value(tf.matmul(mask_M, tf.transpose(self.mat_A)), clip_value_min=0, clip_value_max=1)

    def re_initial(self):
        self.regularize_mat_A(replace=True)
        mask_M = tf.matmul(self.new_mat_A, self.mat_M)
        self.mask = tf.clip_by_value(tf.matmul(mask_M, tf.transpose(self.new_mat_A)), clip_value_min=0, clip_value_max=1)
        self.regularize_mat_B(replace=True)

    def regularize_mat_B(self, replace = False):
        maskM = self.mat_B * self.mask
        maskM = np.clip(maskM, a_min=0, a_max=None)
        self.proMatrix = maskM / np.sum(maskM, 1, keepdims=True)

        if replace:
            self.mat_B = tf.Variable(initial_value=self.proMatrix, trainable=self.train_B, name='Matrix B', dtype=tf.float32)

    def regularize_mat_A(self, replace = False):
        index_max = np.argmax(self.mat_A.numpy(), axis=0)
        new_mat_A = np.zeros_like(self.mat_A.numpy())
        for i, j in enumerate(index_max):
            new_mat_A[j][i] = 1
        self.new_mat_A = new_mat_A

        if replace:
            self.mat_A = tf.Variable(initial_value=new_mat_A, trainable=self.train_A, name='Matrix B', dtype=tf.float32)

############################# calculate the penalty term #############################
    def penalty_1(self, penalty_term = 0.1):
        """
        First penalty
        :return:
        """
        entro = self.entropy(tf.transpose(self.mat_A), tf.transpose(self.mat_A))
        loss = tf.abs(tf.reduce_sum(entro)) * penalty_term

        self.add_loss(loss)

    def penalty_2(self, mat_A_T, penalty_term = 0.1):
        """
        Second penalty
        :return:
        """
        entro = self.entropy(mat_A_T, mat_A_T)
        loss = tf.reduce_sum(entro) * penalty_term

        self.add_loss(loss)

    def penalty_3(self, W, penalty_term = 0.1):
        """
        Third penalty
        :return:
        """
        temp_1 = tf.matmul(self.mat_A, self.mat_M)
        temp_2 = tf.clip_by_value(tf.matmul(temp_1, tf.transpose(self.mat_A)), clip_value_min=0, clip_value_max=1)
        #temp_2 = tf.stop_gradient(temp_2)

        temp_loss = tf.multiply((1 - temp_2), W)
        temp_loss = tf.abs(temp_loss)
        loss = tf.reduce_sum(temp_loss) * penalty_term

        self.add_loss(loss)

    def my_regularizer(self, W, axis=1):
        cumulate = tf.keras.backend.sum(W, axis=axis)
        constant = tf.keras.backend.ones(shape=cumulate.shape, dtype=tf.float32)
        subs = tf.math.subtract(cumulate, constant)

        loss = self.alpha1 * 1 / 2 * tf.reduce_sum(tf.square(subs))
        loss += self.alpha2 * tf.reduce_sum(self.relu(tf.math.negative(W)))
        loss += self.alpha3 * tf.reduce_sum(self.relu(W - 1))

        self.add_loss(loss)

############################# clear state #############################
    def clean_inference(self):
        self.inter_result = None

############################# check for condition #############################
    def check_condition(self):
        check_unique_axis(self.mat_A.numpy())
        check_unique_axis(self.mat_B.numpy(), axis=1)
        check_unique_axis(self.mat_T.numpy(), axis=1)

        a_t = tf.clip_by_value(tf.matmul(self.mat_A, self.mat_T), clip_value_min=0, clip_value_max=1)
        check_unique_axis(a_t.numpy(), axis=1)

        print("No problem!")
