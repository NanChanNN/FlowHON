import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'hon_or')
sys.path.insert(0, 'neuralNet3')
from neuralNet3.nn.dataSetLoader import HoNCluLoader
from neuralNet3.nn.layer import myLayerLinear_V2
from neuralNet3.nn.HoN_v2 import hon_clu
from neuralNet3.nn.trainer import trainer
from auxiliary_tool import *
from check_point import check_point
from neuralNet3.performanceTest import autoTest
from measurment import *
import pandas as pd
import time

def sequence2node(seq):
    filed = seq.strip().split('.')
    nodes = filed[-1] + "|"
    filed.pop()
    for index in range(len(filed)):
        cur_index = len(filed) - index -1
        if cur_index != 0:
            nodes += filed[cur_index] + "."
        else:
            nodes += filed[cur_index]
    return nodes

def get_new_count(new_M, clu_status2index, ref_nodeToIndex, clu_node_sum):
    count_matrix = np.zeros((len(clu_status2index), new_M.shape[1]))
    for status, count_index in clu_status2index.items():
        ref_name = sequence2node(status)
        new_M_index = ref_nodeToIndex[ref_name]
        status_sum = clu_node_sum[status]

        count_matrix[count_index, :] = new_M[new_M_index, :] * status_sum

    return count_matrix

def get_new_counter(new_M, ole_counter, clu_node_sum, clu_name2index, ref_node2index):
    new_counter =  {}
    for source in ole_counter.keys():
        new_counter[source] = {}
        source_name = sequence2node(source)
        source_id = ref_node2index[source_name]
        source_sum = clu_node_sum[source]
        for target in ole_counter[source].keys():
            source_list = source.strip().split('.')
            target_name = ""
            for b_id in range(1, len(source_list)):
                target_name += source_list[b_id] + "."
            target_name += target
            if target == '-1':
                target_id = clu_name2index['-1']
            else:
                target_id = clu_name2index[target_name]

            new_counter[source][target] = new_M[source_id][target_id] * source_sum
    return new_counter

flow = sys.argv[1]
#flow = 'random-5cpR'
#flow = 'computer_room'
my_order = int(sys.argv[2])

change_counter = False

class hp:
    # flow name: random-5cpR, abc, bernard, hurricane_downsample, tornado, two_swirl, crayfish, plume, cylinder, electro3D,
    # lifted_sub, computer_room
    flow_name = flow
    original = False
    train_counter = change_counter
    manipulation = "Linear"
    clu_type = "MSE"
    epoch_per_run = 10
    tolerance = 4
    order = my_order # 3
    mini_support = 1
    maxi_difference = 0.04
    data_document = ".."
    along_time = True
    max_order = my_order # 3
    cluster_size = 10000
    train_id = 2
    test_id = 3

my_columns, my_indices, measure_record = create_record()

data_path = os.path.join(hp.data_document, "Data", hp.flow_name)
_, cluster_training, original_training = get_data_set_name(data_path, hp.flow_name, hp.train_id, hp.order)
_, cluster_validation, original_validation = get_data_set_name(data_path, hp.flow_name, hp.test_id, hp.order)
cluster_validation = original_validation

graph_post, image_folder, image_post = get_file_information(hp.clu_type, hp.manipulation)
out_path_clu = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graphClu.csv")

start_time = time.time()

# original clustering version
clu_gen = hon_clu(cluster_training, order = hp.order,
                  mini_sup = hp.mini_support, maxi_diff = hp.maxi_difference)
clu_gen.generate_hon()

# construct checker
check_loader = HoNCluLoader(hp.flow_name,
                               '',
                               hp.data_document,
                               cluster_training,
                               cluster_validation,
                               clusterSize=hp.cluster_size,
                               setted_model=clu_gen,
                               max_order = hp.order)
ref_data_loader = HoNCluLoader(hp.flow_name,
                               'Ref',
                               hp.data_document,
                               cluster_training,
                               cluster_validation,
                               clusterSize=hp.cluster_size,
                               max_order = hp.order)

checker = check_point(data_loader=check_loader, manipulation=hp.manipulation)

# check original version
checker.check(clu_gen)

# construct trainer
CluTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Clustering', data_loader=check_loader)
CluTrainer.run()

# initialize trained matrix
trainedT = CluTrainer.get_transition_mat()

# check trained original version
checker.check(clu_gen, CluTrainer.get_transition_mat())

epoch = 1

while not checker.exceed_tolerance(hp.tolerance):
    epoch += 1
    # Step 2
    T = clu_gen.generate_trainsition_map(block_map=ref_data_loader.regionIndex)
    M, M_mask = clu_gen.generate_status_node_transition(status_map=ref_data_loader.nodeToIndex)
    s_n_map= clu_gen.get_status_node_map(status_map=ref_data_loader.nodeToIndex)
    linear_layer = myLayerLinear_V2(ref_data_loader.vertexNumber, ref_data_loader.regionNumber,
                                 wM=M,
                                 tM=T,
                                 mM=M_mask, )
    my_trainer = trainer(flowName=hp.flow_name,
                         manipulation=hp.manipulation,
                         graphType='Ref',
                         data_loader=ref_data_loader,
                         training_model=linear_layer)
    my_trainer.run(hp.epoch_per_run, s_n_map)

    # Step 3
    new_M = my_trainer.get_transition_mat()
    # version 2 modification
    new_M = manipulate_matrix(new_M, hp.manipulation, M_mask)

    new_count = get_new_count(new_M, clu_gen.status2index, ref_data_loader.nodeToIndex, clu_gen.node_sum)
    clu_gen.new_pro_matrix = new_count
    if hp.train_counter:
        new_counter = get_new_counter(new_M, clu_gen.counter, clu_gen.node_sum, clu_gen.name2index,
                                      ref_data_loader.nodeToIndex)
        clu_gen.counter = new_counter

    if hp.original:
        clu_gen.generate_hon()
    else:
        clu_gen.generate_hon_wrt_T(trainedT)
    # checkpoint
    checker.check(clu_gen)

    # Step 4
    check_loader.rerun = True
    check_loader.run(hon_network=clu_gen)
    tmp_trainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Clustering',
                          data_loader=check_loader)
    tmp_trainer.run()

    trainedT = tmp_trainer.get_transition_mat()

    # checkpoint
    checker.check(clu_gen, tmp_trainer.get_transition_mat())


result_time = time.time() - start_time
my_columns.append('Time')
my_columns.append('Ref')
my_columns.append('Ref+')
my_columns.append('epoch')
measure_record['epoch'] = [epoch]
measure_record['Time'] = [result_time]
measure_record['Ref'] = []
measure_record['Ref+'] = []

result = checker.best_result
#print(checker.max_index)

# Previous result
performance = autoTest(hp.flow_name, hp.data_document, original_training, cluster_training, original_validation, cluster_validation,
                       clusterSize=hp.cluster_size, cluster_file_path=out_path_clu, max_order=hp.order,
                       alongTime=hp.along_time)
#performance.showTestResult()

# get reference result
reference_result = performance.initialResult['Ref']
# Construct graph edge tunner

FoNTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='FoN', data_loader=performance.FoN)
SemTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Semantic', data_loader=performance.Semantic)
RefTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Ref', data_loader=performance.Ref)

# edge tunning

FoNTrainer.run()
SemTrainer.run()
RefTrainer.run()

# update random walk result

performance.randomWalkTestSemantic(SemTrainer.get_transition_mat(), hp.manipulation)
performance.randomWalkTestFoN(FoNTrainer.get_transition_mat(), hp.manipulation)
performance.randomWalkTestRef(RefTrainer.get_transition_mat(), hp.manipulation)
performance.result['Clu'] = result

# make measurement record
kld, proport = measure(reference_result, performance.initialResult['FoN'])
measure_record['FoN'].extend([kld, proport])
kld, proport = measure(reference_result, performance.result['FoN'])
measure_record['FoN with edge training'].extend([kld, proport])
kld, proport = measure(reference_result, performance.initialResult['Semantic'])
measure_record['Semantic'].extend([kld, proport])
kld, proport = measure(reference_result, performance.result['Semantic'])
measure_record['Semantic with edge training'].extend([kld, proport])
kld, proport = measure(reference_result, performance.initialResult['Ref'])
measure_record['Ref'].extend([kld, proport])
kld, proport = measure(reference_result, performance.result['Ref'])
measure_record['Ref+'].extend([kld, proport])
kld, proport = measure(reference_result, checker.result_log[0])
measure_record['Original Clu'].extend([kld, proport])
kld, proport = measure(reference_result, checker.best_result)
measure_record['Clu'].extend([kld, proport])
#np.testing.assert_almost_equal(checker.result_log[0], performance.initialResult['Clu'])
save_path = os.path.join(data_path, 'experiment_record'+str(hp.order)+'.csv')
data_frame = pd.DataFrame(measure_record, columns=my_columns, index=my_indices)
data_frame.to_csv(path_or_buf=save_path)


tmp_path = os.path.join(hp.data_document, "Result_New", "two_step_v2")
result_file = os.path.join(tmp_path, hp.flow_name)
performance.showTestResult_v3(initialRes = ['FoN', 'Semantic', 'Ref'],
                           manipulation = hp.manipulation,
                           save_path=result_file)

out_weight_fon = os.path.join(data_path, "TrainedWeight-1.csv")
performance.store_matrix_weights(matrix=FoNTrainer.get_transition_mat(),
                                 manipulate=hp.manipulation,
                                 mask=performance.FoN.getMask(),
                                 graph=performance.FoN.graph,
                                 output_file=out_weight_fon,)

out_weight_semantic = os.path.join(data_path, "TrainedWeight-"+str(hp.order)+".csv")
performance.store_matrix_weights(matrix=SemTrainer.get_transition_mat(),
                                 manipulate=hp.manipulation,
                                 mask=performance.Semantic.getMask(),
                                 graph=performance.Semantic.graph,
                                 output_file=out_weight_semantic,)
out_weight_clustering = os.path.join(data_path, "TrainedWeight-Clu-"+str(hp.order)+".csv")
performance.store_matrix_weights(matrix=checker.transition_matrix,
                                 manipulate=hp.manipulation,
                                 mask=checker.mask,
                                 graph=checker.HoNClu.graph_edges,
                                 output_file=out_weight_clustering,)
out_graph_clu_path = os.path.join(data_path, hp.flow_name + str(hp.order) + '-graphClu.csv')
checker.HoNClu.load_graph_to_file(out_graph_clu_path)

















