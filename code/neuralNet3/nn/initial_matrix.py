import numpy as np
from nn.dataSetLoader import HoNCluLoader

class initialer:
    def __init__(self, flow_name):
        self.HoNClu = HoNCluLoader(flow_name, '')
        self.HoNRef = HoNCluLoader(flow_name, 'Ref')

        self.construct_mat_A()
        self.construct_mat_B()
        self.construct_mat_T()
        self.construct_mat_M()

        self.check_all_condition()

    def construct_mat_A(self):
        self.m = self.HoNClu.vertexNumber
        self.n = self.HoNRef.vertexNumber
        self.mat_A = np.zeros((self.m, self.n))
        for key, j in self.HoNRef.nodeToIndex.items():
            i = self.HoNClu.nodeToIndex[key]
            self.mat_A[i][j] = 1

    def construct_mat_B(self):
        self.mat_B = self.HoNClu.getTransitionProb()

    def construct_mat_T(self):
        self.mat_T = self.HoNRef.mapMatrix

    def construct_mat_M(self):
        self.mat_M = self.HoNRef.getMask()

    # debug function
    def check_unique(self):
        size = self.HoNRef.vertexNumber
        counter = np.zeros(size)
        for _, item in self.HoNRef.nodeToIndex.items():
            counter[item] += 1
        assert np.all(counter == 1)

    def check_unique_axis(self, mat, axis=0):
        sum_col = np.sum(mat, axis=axis)
        bool_col = np.abs(sum_col - 1.0) < 1e-05
        assert np.all(bool_col) == True

    def check_all_condition(self):
        """
        Check whether the condition satisfy.
        :return:
        """
        self.check_unique()
        self.check_unique_axis(self.mat_A)
        self.check_unique_axis(self.mat_B, axis=1)
        self.check_unique_axis(self.mat_T, axis=1)

        assert (self.mat_M.shape == (self.n, self.n))

        mask_M = np.matmul(self.mat_A, self.mat_M)
        mask_ref = (np.matmul(mask_M, np.transpose(self.mat_A)) > 0)
        mask_ref = mask_ref.astype(np.int32)
        mask_clu = self.HoNClu.getMask()
        assert  mask_clu.shape == mask_ref.shape
        assert  np.all(mask_clu==mask_ref)

        count_ref = np.matmul(self.mat_A, self.HoNRef.transitionMatrix_count)
        count_ref = np.matmul(count_ref, np.transpose(self.mat_A))
        count_clu = self.HoNClu.transitionMatrix_count
        assert count_ref.shape == count_clu.shape
        assert np.all(count_ref==count_clu)

        map_matrix_clu = self.HoNClu.mapMatrix
        map_matrix_ref = np.clip(np.matmul(self.mat_A, self.mat_T), 0, 1)
        assert map_matrix_clu.shape == map_matrix_ref.shape
        assert np.all(map_matrix_ref==map_matrix_clu)

        print("check condition: OK!")
        return True


