import os
import csv
import pandas as pd
import numpy as np

def get_data_set_name(path_docu, flow_name, size, order=3):
    if size == 0:
        ret_path = os.path.join(path_docu, flow_name + "-DataSequen-Small.csv")
    elif size == 1:
        ret_path = os.path.join(path_docu, flow_name + "-DataSequen-Normal.csv")
    elif size == 2:
        ret_path = os.path.join(path_docu, flow_name + "-DataSequen-Large.csv")
    else:
        ret_path = os.path.join(path_docu, flow_name + "-DataSequen-Extreme.csv")

    order_postfix = ''
    if order != 3:
        order_postfix = "-order" + str(order)

    return ret_path[:-4] + "-semantic" + order_postfix + ".csv", \
           ret_path[:-4] + "-clu" + order_postfix + ".csv", \
           ret_path[:-4] + "-original" + order_postfix + ".csv"

def pre_process_file(file, order = 3):
    df = pd.read_csv(file, header=None, names=['seq', 'seed_pos', 'seeds'])
    new_sequence = []
    org_sequence = []
    order_postfix = ''
    if order != 3:
        order_postfix = '-order' + str(order)

    out_file_path_1 = file[:-4] + "-clu" + order_postfix + ".csv"
    out_file_path_2 = file[:-4] + "-semantic" + order_postfix + ".csv"
    out_file_path_3 = file[:-4] + "-original" + order_postfix + ".csv"
    if os.path.exists(out_file_path_1) and os.path.exists(out_file_path_2) and os.path.exists(out_file_path_3) and False:
        return

    #order = 4
    for a in df.to_numpy():
        #new_seq = [-1, -1]
        new_seq = [-1 for _ in range(order-1)]
        seed_pos = int(a[1])
        seq = a[0].strip().split(' ')
        new_seq.extend([int(n) for n in seq[1:]])
        new_seq = new_seq[seed_pos - 1:]
        org_sequence.append([int(n) for n in seq[seed_pos:]])
        new_sequence.append(new_seq)

    if not os.path.exists(out_file_path_3) or True:
        with open(out_file_path_3, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, seq in enumerate(org_sequence):
                result = str(id)
                for n in seq:
                    result += (' ' + str(n))
                writer.writerow([result])

    if not os.path.exists(out_file_path_1) or True:
        with open(out_file_path_1, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, seq in enumerate(new_sequence):
                result = str(id)
                for n in seq:
                    result += (' ' + str(n))
                writer.writerow([result])

    if not os.path.exists(out_file_path_2) or True:
        with open(out_file_path_2, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, seq in enumerate(new_sequence):
                result = str(id)
                for n in seq:
                    if n == -1:
                        continue
                    result += (' ' + str(n))
                writer.writerow([result])

def pre_process_rawdata(path_docu, flow_name, order=3):
    postfix = ['Small', 'Extreme', 'Large', 'Normal']
    for post in postfix:
        file = os.path.join(path_docu, flow_name + "-DataSequen-"+ post + ".csv")
        pre_process_file(file, order)

def get_file_information(clu_type, manipulation):
    # "-graphClu-MSE.csv"
    # "-graphClu-Cov.csv"
    # "-graphClu.csv"
    assert clu_type in ["MSE", "Cos", "KLD"]
    image_folder = ""
    image_post = ""
    graph_post = ""
    if clu_type == "MSE":
        graph_post = "-graphClu-MSE.csv"
        image_folder = "MSE"
        image_post = "M"
    elif clu_type == "Cos":
        graph_post = "-graphClu-Cov.csv"
        image_folder = "cosine"
        image_post = ""
    elif clu_type == "KLD":
        graph_post = "-graphClu.csv"
        image_folder = "KLD"
        image_post = "K"

    if manipulation == "Division":
        image_folder += "_Div"

    return graph_post, image_folder, image_post

def get_order_postfix(order):
    if order == 3:
        return ''
    else:
        return '-o' + str(order)

def create_record():
    columns = ['FoN', 'FoN with edge training',
               'Semantic', 'Semantic with edge training',
               'Original Clu', 'Clu']
    index = ['avg KLD', 'avg proportion']
    data_frame = {}
    for name in columns:
        data_frame[name] = []

    return columns, index, data_frame

def manipulate_matrix(matrix, manipulate, mask):
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

def load_trained_weight(file, size):
    return_mat = np.zeros(size)
    with open(file) as f:
        for ele in f:
            fields = ele.strip().split(',')
            source = int(fields[0])
            target = int(fields[1])
            probability = float(fields[2])
            return_mat[source][target] = probability
    return  return_mat
