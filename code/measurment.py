import numpy as np

def average(input_array):
    return np.mean(input_array)

def avg_of_proportion(ref_array, input_array):
    tmp_array = input_array / ref_array
    return average(tmp_array)

def measure(ref_array, input_array):
    ref_array = np.array(ref_array).reshape(-1)
    input_array = np.array(input_array).reshape(-1)
    r_1 = average(input_array)
    r_2 = avg_of_proportion(ref_array, input_array)
    return r_1, r_2