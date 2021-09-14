import sys
sys.path.insert(0, 'hon_or')
sys.path.insert(0, 'neuralNet3')
import neuralNet3.nn.dataSetLoader
from hon_or.main_hon import top_layer
from neuralNet3.nn.trainer import trainer
from neuralNet3.nn.HoN_v2 import hon_clu
from neuralNet3.performanceTest import autoTest
from auxiliary_tool import *
import os

flow = sys.argv[1]
my_order = int(sys.argv[2])
cluster_type = "MSE"#sys.argv[2]
manipulate_type = "Linear"#sys.argv[3]
#max_order = int(sys.argv[4])
#al_time = bool(int(sys.argv[4])==1)

class hp:
    # flow name: random-5cpR, abc, bernard, hurricane_downsample, tornado, two_swirl, crayfish, plume, cylinder, electro3D,
    # lifted_sub, computer_room
    flow_name = flow
    order = 1
    max_order = my_order#2
    mini_support = 1
    maxi_difference = 0.04
    data_document = ".."
    manipulation = manipulate_type#manipulation = "Linear"
    clu_type = cluster_type #clu_type = "MSE"
    cluster_size = 10000#1000
    rerun = False
    along_time = True
    train_id = 2
    test_id = 3

data_path = os.path.join(hp.data_document, "Data", hp.flow_name)

#pre_process_rawdata(data_path, hp.flow_name, hp.max_order)
semantic_training, cluster_training, original_training = get_data_set_name(data_path, hp.flow_name, hp.train_id, hp.max_order)
semantic_validation, cluster_validation, original_validation = get_data_set_name(data_path, hp.flow_name, hp.test_id, hp.max_order)
semantic_training = original_training
semantic_validation = original_validation
cluster_validation = original_validation
print(semantic_training)

# get max_order postfix
max_order_post = get_order_postfix(hp.max_order)
max_order_post = ''

# generate first-order network
hp.order = 1
semantic_1_file = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graph" + ".csv")
if not os.path.exists(semantic_1_file) or hp.rerun:
    #top_layer(hp.order, hp.mini_support, hp.flow_name, hp.data_document, training_file)
    top_layer(hp.order, hp.mini_support, hp.flow_name, hp.data_document, semantic_training, semantic_1_file)

# generate third-order network
hp.order = hp.max_order
semantic_3_file = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graph" + ".csv")
if not os.path.exists(semantic_3_file) or hp.rerun:
    #top_layer(hp.order, hp.mini_support, hp.flow_name, hp.data_document, training_file)
    top_layer(hp.order, hp.mini_support, hp.flow_name, hp.data_document, semantic_training, semantic_3_file)

difference_path = os.path.join(hp.data_document, "Result", "KLD_Diff")
difference_path = os.path.join(difference_path, hp.flow_name + "KLD_Diff")
hp.maxi_difference = sys.float_info.max
clu_gen = hon_clu(cluster_training, order = hp.order, mini_sup = hp.mini_support, maxi_diff = hp.maxi_difference)
clu_gen.generate_hon()
diff, bins = clu_gen.show_difference(title=hp.flow_name + " difference distribution",
save_path=None, critical_point=0.04)

ret_list = list(bins[0])
with open(difference_path+".csv", "w", newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(ret_list)

#print(diff)
#print(bins)
exit(0)

graph_post, image_folder, image_post = get_file_information(hp.clu_type, hp.manipulation)

# generate clustering network
#out_path_clu = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graphClu-MSE.csv")
out_path_clu = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graphClu.csv")
if not os.path.exists(out_path_clu) or hp.rerun:
#if os.path.exists(out_path_clu):
    #clu_gen = hon_clu(training_file, order = hp.order, mini_sup = hp.mini_support, maxi_diff = hp.maxi_difference)
    clu_gen = hon_clu(cluster_training, order = hp.order, mini_sup = hp.mini_support,
                      maxi_diff = hp.maxi_difference, distance_metric=hp.clu_type)
    clu_gen.generate_hon(out_path_clu)

# generate reference network
hp.maxi_difference = -sys.float_info.max
out_path_ref = os.path.join(data_path, hp.flow_name + str(hp.order) + "-graphRef.csv")
if not os.path.exists(out_path_ref) or hp.rerun:
    #clu_gen = hon_clu(training_file, order = hp.order, mini_sup = hp.mini_support, maxi_diff = hp.maxi_difference)
    clu_gen = hon_clu(cluster_training, order=hp.order, mini_sup=hp.mini_support, maxi_diff=hp.maxi_difference)
    clu_gen.generate_hon(out_path_ref)

# training stage
#performance = autoTest(hp.flow_name, hp.data_document, training_file, training_file, validation_file, validation_file, clusterSize=hp.cluster_size)
#performance = autoTest(hp.flow_name, hp.data_document, semantic_training, cluster_training, semantic_validation, cluster_validation,
#                       clusterSize=hp.cluster_size, cluster_file_path=out_path_clu, max_order=hp.max_order,
#                       alongTime=hp.along_time)
performance = autoTest(hp.flow_name, hp.data_document,
                       train_file_seman = semantic_training,
                       train_file_clu = cluster_training,
                       valid_file_semantic = original_validation,
                       valid_file_clu = original_validation,
                       clusterSize=hp.cluster_size, cluster_file_path=out_path_clu, max_order=hp.max_order,
                       alongTime=hp.along_time)
#performance.showTestResult()

# Construct graph edge tunner

FoNTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='FoN', data_loader=performance.FoN)
SemTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Semantic', data_loader=performance.Semantic)
CluTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Clustering', data_loader=performance.Clustering)
RefTrainer = trainer(flowName=hp.flow_name, manipulation=hp.manipulation, graphType='Ref', data_loader=performance.Ref)

# edge tunning

FoNTrainer.run(100)
SemTrainer.run(100)
CluTrainer.run(100)
RefTrainer.run(100)

'''
fon_weighted_file = os.path.join(data_path, "TrainedWeight-1.csv")
fon_trained_weight = load_trained_weight(file=fon_weighted_file, size=performance.FoN.transitionMatrix.shape)
semantic_weighted_file = os.path.join(data_path, "TrainedWeight-"+str(hp.order)+".csv")
semantic_trained_weight = load_trained_weight(file=semantic_weighted_file, size=performance.Semantic.transitionMatrix.shape)
clu_weighted_file = os.path.join(data_path, "TrainedWeight-Clu-"+str(hp.order)+".csv")
clu_trained_weight = load_trained_weight(file=clu_weighted_file, size=performance.Clustering.transitionMatrix.shape)
#print(performance.Clustering.transitionMatrix.shape)
# update random walk result
'''

#performance.randomWalkTestSemantic(semantic_trained_weight, hp.manipulation)
performance.randomWalkTestSemantic(SemTrainer.get_transition_mat(), hp.manipulation)
#performance.randomWalkTestClu(clu_trained_weight, hp.manipulation)
performance.randomWalkTestClu(CluTrainer.get_transition_mat(), hp.manipulation)
#performance.randomWalkTestFoN(fon_trained_weight, hp.manipulation)
performance.randomWalkTestFoN(FoNTrainer.get_transition_mat(), hp.manipulation)
performance.randomWalkTestRef(RefTrainer.get_transition_mat(), hp.manipulation)

#tmp_path = os.path.join(hp.data_document, "Result", "MSE")
'''
result_folder = "Result_New"
if hp.max_order != 3:
    result_folder += str(hp.max_order)
tmp_path = os.path.join(hp.data_document, result_folder, image_folder)
a_post = ''
if hp.along_time:
    a_post ='-A'
result_file = os.path.join(tmp_path, hp.flow_name + max_order_post +
                           "RW-"+str(hp.train_id)+"-"+str(hp.test_id)+
                           hp.manipulation+image_post+a_post)
'''

tmp_path = os.path.join(hp.data_document, "Result_New", "Test"+str(hp.max_order))
result_file = os.path.join(tmp_path, hp.flow_name)
performance.showTestResult(initialRes = ['FoN', 'Semantic', 'Clu', 'Ref'],
                           manipulation = hp.manipulation,
                           save_path=result_file)
'''
import matplotlib.pyplot as plt
x = range(len(CluTrainer.trainingRecord['training loss record']))
plt.plot(x, CluTrainer.trainingRecord['training loss record'])
plt.show()
'''


'''
out_weight_fon = os.path.join(data_path, "TrainedWeight-1.csv")
performance.store_matrix_weights(matrix=FoNTrainer.get_transition_mat(),
                                 manipulate=hp.manipulation,
                                 mask=performance.FoN.getMask(),
                                 graph=performance.FoN.graph,
                                 output_file=out_weight_fon,)
out_weight_semantic = os.path.join(data_path, "TrainedWeight-3.csv")
performance.store_matrix_weights(matrix=SemTrainer.get_transition_mat(),
                                 manipulate=hp.manipulation,
                                 mask=performance.Semantic.getMask(),
                                 graph=performance.Semantic.graph,
                                 output_file=out_weight_semantic,)
out_weight_clustering = os.path.join(data_path, "TrainedWeight-Clu-3.csv")
performance.store_matrix_weights(matrix=CluTrainer.get_transition_mat(),
                                 manipulate=hp.manipulation,
                                 mask=performance.Clustering.getMask(),
                                 graph=performance.Clustering.graph,
                                 output_file=out_weight_clustering,)
'''