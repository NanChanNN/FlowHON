import sys
import numpy as np
import copy

class check_point:

    def __init__(self, data_loader, manipulation = 'Linear'):
        self.data_loader = copy.deepcopy(data_loader)
        self.HoNClu = None
        self.transition_matrix = None
        self.best_result = None
        self.mask = None
        self.result_log = []
        self.minimum_value = sys.float_info.max
        self.manipulation = manipulation
        self.still = 0
        self.max_index = -1

    def check(self, hon, tran_matrix = None):
        self.data_loader.rerun = True
        self.data_loader.run(hon_network=hon)

        if tran_matrix is None:
            tran_matrix = self.data_loader.getTransitionProb()

        tmp_mask = self.data_loader.getMask()
        pro_matrix = self. manipulateMarix(tran_matrix, self.manipulation, tmp_mask)
        result = self.data_loader.randomWalkTest(pro_matrix)

        avg_kld = np.mean(result)
        if avg_kld < self.minimum_value:
            self.minimum_value = avg_kld
            self.best_result = result
            self.still = 0

            self.HoNClu = copy.deepcopy(hon)
            self.transition_matrix = copy.deepcopy(tran_matrix)
            self.mask = copy.deepcopy(tmp_mask)
            self.max_index = len(self.result_log)
        else:
            self.still += 1
        self.result_log.append(result)

        return result

    def manipulateMarix(self, matrix, manipulate, mask):
        proMatrix = None
        if (manipulate == "Division"):
            maskM = matrix * mask
            absoluteW = np.abs(maskM)
            proMatrix = absoluteW / np.sum(absoluteW, 1, keepdims=True)
        elif (manipulate == "Softmax"):
            expW = np.exp(matrix) * mask
            proMatrix = expW / np.sum(expW, 1, keepdims=True)
        elif (manipulate == "Linear"):
            maskM = matrix * mask
            maskM = np.clip(maskM, a_min=0, a_max=None)
            proMatrix = maskM / np.sum(maskM, 1, keepdims=True)
        return proMatrix

    def exceed_tolerance(self, tolerance):
        return self.still > tolerance

