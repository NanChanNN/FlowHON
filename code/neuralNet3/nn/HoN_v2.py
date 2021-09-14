import numpy as np
from nn.utls import *
import sys
import itertools
import csv
import time
import copy
import matplotlib.pyplot as plt
from scipy.spatial import distance


class hon_clu:
############### initialize input information ################
    def __init__(self, data_path, order = 3, mini_sup = 1, maxi_diff = 0.5, verbose=0, distance_metric='MSE'):
        self.verbose = verbose
        self.order = order
        self.mini_sup = mini_sup
        self.maxi_diff = maxi_diff
        self.sequence = self.read_raw_data(data_path)
        self.counter = {}
        self.distribution = {}
        self.node_sum = {}
        self.region_info = {}
        self.status2index = {}
        self.block2index = {}
        self.block_cluster = {}
        self.tran_matrix = None
        self.count_matrix = None
        self.global_difference = []
        self.new_pro_matrix = None
        self.d_metric = distance_metric
        assert self.d_metric in ['Cos', 'KLD', 'MSE']
        self.record_list = None

        self.construct_status()
        self.construct_tran_matrix()

    def construct_status(self):
        subseq_list = []
        for seq in self.sequence:
            subseq_list.extend(self.extract_sub_sequence(seq, self.order))

        for sub_seq in subseq_list:
            target = sub_seq[-1]
            source = self.all_but_last_element(sub_seq)
            self.increase_dict(self.counter, source, target)

        # build the distribution
        self.check_for_mini_sup()
        for source in self.counter.keys():
            s_sum = sum(self.counter[source].values())
            if s_sum == 0:
                continue
            for target, value in self.counter[source].items():
                self.increase_dict(self.distribution, source, target, float(value)/float(s_sum))
            self.node_sum[source] = s_sum
        self.node_sum['-1'] = 0
        self.build_region_info()

    def extract_sub_sequence(self, sequence, order):
        subseq = []
        length = order+1
        if len(sequence) < length:
            return []
        for index in range(len(sequence) - length + 1):
            subseq.append(sequence[index:index+length])
        return subseq

    def build_region_info(self):
        for node_name in self.distribution.keys():
            region_name = self.last_element(node_name)
            self.initial_dict(self.region_info, region_name, [])
            self.region_info[region_name].append(node_name)
        self.region_info['-1'] = []

    def construct_tran_matrix(self):
        block_size = len(self.region_info.keys())
        for id, b in enumerate(self.region_info.keys()):
            self.block2index[b] = id
        node_size = len(self.distribution.keys())
        for id, n in enumerate(self.distribution.keys()):
            self.status2index[n] = id
        self.tran_matrix = np.zeros((node_size, block_size))
        self.count_matrix = np.zeros((node_size, block_size))

        for source in self.distribution.keys():
            source_id = self.status2index[source]
            for target, pro in self.distribution[source].items():
                target_id = self.block2index[target]
                self.tran_matrix[source_id][target_id] = pro
                self.count_matrix[source_id][target_id] = self.counter[source][target]
        check_unique_axis(self.tran_matrix, axis=1)

    def get_initial_distribute(self, file_path=None):
        record_list = [0 for _ in range(len(self.graph_nodes))]
        for seq in self.sequence:
            name = seq[:self.order+1]
            name = self.all_but_last_element(name)
            index = self.name2index[name]
            record_list[index] += 1

        self.record_list = record_list
        if file_path is not None:
            out_path = file_path[:-4] + "-intial.csv"
            with open(out_path, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                for id, count in enumerate(record_list):
                    writer.writerow([id, count])

############### generate hon ################
    def generate_hon(self, file_path=None):
        self.display("Generate clusters ...")
        self.generate_clusters()

        self.display("Generating edges ...")
        self.construct_edges()

        self.get_initial_distribute(file_path)
        if file_path is not None:
            self.display("Loading graph ...")
            self.load_graph_to_file(file_path)
    
    def generate_hon_wrt_T(self, mT, file_path=None):
        self.display("Generate clusters ...")
        self.generate_clusters_wrt_T(mT)

        self.display("Generating edges ...")
        self.construct_edges()

        self.get_initial_distribute(file_path)
        if file_path is not None:
            self.display("Loading graph ...")
            self.load_graph_to_file(file_path)


    def load_graph_to_file(self, output_path):
        with open(output_path,"w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.block_num, self.node_num])
            for block, index in self.block2index.items():
                writer.writerow([block, index])
            for id, nodes in enumerate(self.graph_nodes):
                block = self.last_element(nodes[0])
                write_row = [block]
                for n in nodes:
                    write_row.append(self.sequence2node(n))
                write_row.append(str(self.clusters_weights[id]))
                writer.writerow(write_row)
            for source_id in self.graph_edges.keys():
                for target_id, value in self.graph_edges[source_id].items():
                    writer.writerow([source_id, target_id, value])

    def sequence2node(self, seq):
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

############### hierarchical clustering ################
    def generate_clusters(self):
        self.graph_nodes = []
        self.name2index = {}
        self.region_cluster = {}

        for region, status in self.region_info.items():
            clusters = self.hierarchical_clustering(status)
            self.add_to_nodes(clusters, region)

        self.initial_dict(self.region_cluster, '-1', [])
        self.name2index['-1'] = len(self.graph_nodes)
        self.graph_nodes.append(['-1'])
        self.calculate_clu_weight()

        self.block_num = len(self.block2index.keys())
        self.node_num = len(self.graph_nodes)

    def generate_clusters_wrt_T(self, mT):
        tmp_region_cluster = copy.deepcopy(self.region_cluster)

        self.graph_nodes = []
        self.name2index = {}
        self.region_cluster = {}

        for region, status in self.region_info.items():
            #clusters = self.hierarchical_clustering(status)
            clusters = self.clustering_based_on_T(status, tmp_region_cluster[region], mT)
            self.add_to_nodes(clusters, region)

        self.initial_dict(self.region_cluster, '-1', [])
        self.name2index['-1'] = len(self.graph_nodes)
        self.graph_nodes.append(['-1'])
        self.calculate_clu_weight()

        self.block_num = len(self.block2index.keys())
        self.node_num = len(self.graph_nodes)

    def hierarchical_clustering(self, status_list):
        clusters = []
        for status in status_list:
            assert len(status.strip().split('.')) == self.order
            clusters.append([status])
        while len(clusters) > 1:
            id1, id2, diff = self.getMinimumCluster(clusters)
            self.global_difference.append(diff)
            if diff > self.maxi_diff:
                break
            else:
                clusters[id1].extend(clusters[id2])
                clusters.pop(id2)
        return clusters

    def clustering_based_on_T(self, status_list, node_list, T):
        clusters = []
        ret_cluster = []
        node_tran_prob = []
        for n_id in node_list:
            clusters.append([])
            node_tran_prob.append(T[n_id, :])

        for status in status_list:
            s_id = self.status2index[status]
            status_tran = self.new_pro_matrix[s_id, :] / self.node_sum[status]
            tmp_result = []
            for n_tran in node_tran_prob:
                dist = np.sqrt(np.square(np.subtract(status_tran, n_tran)).mean())
                tmp_result.append(dist)
            mini_node_id = np.argmin(tmp_result)
            clusters[mini_node_id].append(status)

        for cluster in clusters:
            if len(cluster) != 0:
                ret_cluster.append(cluster)

        return ret_cluster

    def getMinimumCluster(self, clusters):
        id1 = -1
        id2 = -1
        diff = sys.float_info.max
        cluster_size = len(clusters)
        for i in range(cluster_size):
            for j in range(i+1, cluster_size):
                #tmp_difference = self.calculateWeightedLinkage(clusters[i], clusters[j])
                #tmp_difference = self.calculateWeightedLinkage_Vec(clusters[i], clusters[j])
                tmp_difference = self.calculateWeightedLinkage_new(clusters[i], clusters[j])
                #np.testing.assert_almost_equal(tmp_difference, tmp_difference_1)

                if tmp_difference < diff:
                    id1 = i
                    id2 = j
                    diff = tmp_difference

        return id1, id2, diff

    def calculateWeightedLinkage(self, clusters_i, clusters_j):
        sum1 = 0.0
        sum2 = 0.0
        aveSum = 0.0

        for status in clusters_i:
            sum1 += np.sum(self.count_matrix[self.status2index[status], :])
        for status in clusters_j:
            sum2 += np.sum(self.count_matrix[self.status2index[status], :])

        for status_1 in clusters_i:
            status_1_index = self.status2index[status_1]
            status_1_sum = np.sum(self.count_matrix[status_1_index, :])
            status_1_dist = self.count_matrix[status_1_index, :] / status_1_sum

            for status_2 in clusters_j:
                status_2_index = self.status2index[status_2]
                status_2_sum = np.sum(self.count_matrix[status_2_index, :])
                merge_dist = self.count_matrix[status_1_index, :] + self.count_matrix[status_2_index, :]
                merge_dist = merge_dist / np.sum(merge_dist)
                status_2_dist = self.count_matrix[status_2_index, :] / status_2_sum

                if self.d_metric == 'KLD':
                    kld_1 = self.KLD(status_1_dist, merge_dist)
                    kld_2 = self.KLD(status_2_dist, merge_dist)
                    dist = 0.5 * (kld_1 + kld_2)
                elif self.d_metric == "MSE":
                    dist = np.sqrt(np.square(np.subtract(status_1_dist, status_2_dist)).mean())
                else:
                    dist = self.cosine_dist(status_1_dist, status_2_dist)

                aveSum += (status_1_sum / sum1) * (status_2_sum / sum2) * dist

        return aveSum

    def calculateWeightedLinkage_new(self, clusters_i, clusters_j):
        sum1 = 0.0
        sum2 = 0.0
        aveSum = 0.0

        if self.new_pro_matrix is None:
            my_matrix = self.count_matrix
        else:
            my_matrix = self.new_pro_matrix
            print("new pro matrix")

        for status in clusters_i:
            sum1 += np.sum(my_matrix[self.status2index[status], :])
        for status in clusters_j:
            sum2 += np.sum(my_matrix[self.status2index[status], :])

        for status_1 in clusters_i:
            status_1_index = self.status2index[status_1]
            status_1_sum = np.sum(my_matrix[status_1_index, :])
            status_1_dist = my_matrix[status_1_index, :] / status_1_sum

            for status_2 in clusters_j:
                status_2_index = self.status2index[status_2]
                status_2_sum = np.sum(my_matrix[status_2_index, :])
                merge_dist = my_matrix[status_1_index, :] + my_matrix[status_2_index, :]
                merge_dist = merge_dist / np.sum(merge_dist)
                status_2_dist = my_matrix[status_2_index, :] / status_2_sum

                #kld_1 = self.KLD(status_1_dist, merge_dist)
                #kld_2 = self.KLD(status_2_dist, merge_dist)
                #dist = 0.5 * (kld_1 + kld_2)

                dist = np.sqrt(np.square(np.subtract(status_1_dist, status_2_dist)).mean())

                #dist = self.cosine_dist(status_1_dist, status_2_dist)

                aveSum += (status_1_sum / sum1) * (status_2_sum / sum2) * dist

        return aveSum

    def calculateWeightedLinkage_Vec(self, clusters_i, clusters_j):
        aveSum = 0.0
        index_array = [self.status2index[status_2] for status_2 in clusters_j]
        count_array = self.count_matrix[index_array, :]
        sum_per_row = np.sum(count_array, axis=1, keepdims=True)
        sum_per_row_nor = sum_per_row / np.sum(sum_per_row)
        count_array_nor = count_array / sum_per_row

        index_array_1 = [self.status2index[status_1] for status_1 in clusters_i]
        count_array_1 = self.count_matrix[index_array_1, :]
        sum_per_row_1 = np.sum(count_array_1, axis=1, keepdims=True)
        count_array_1_nor = count_array_1 / sum_per_row_1
        sum_per_row_nor_1 = sum_per_row_1 / np.sum(sum_per_row_1)

        size_i = len(clusters_i)

        for index_i in range(size_i):
            status_1_dist = count_array_1_nor[index_i, :]

            merge_array = count_array + count_array_1[index_i, :]
            merge_array_nor = merge_array / np.sum(merge_array, axis=1, keepdims=True)
            kld_1 = self.KLD_high_dimen(status_1_dist, merge_array_nor)
            kld_2 = self.KLD_high_dimen(count_array_nor, merge_array_nor)
            dist = 0.5 * (kld_1 + kld_2)
            aveSum += np.sum(dist * sum_per_row_nor * sum_per_row_nor_1[index_i])

        return aveSum

    def add_to_nodes(self, clusters, region):
        self.initial_dict(self.region_cluster, region, [])
        for nodes in clusters:
            nodes_index = len(self.graph_nodes)
            self.graph_nodes.append(nodes)
            self.region_cluster[region].append(nodes_index)
            for n in nodes:
                self.name2index[n] = nodes_index

    def calculate_clu_weight(self):
        self.clusters_weights = [0 for _ in range(len(self.graph_nodes))]
        for id, nodes in enumerate(self.graph_nodes):
            for n in nodes:
                self.clusters_weights[id] += self.node_sum[n]

        self.regions_weights = {}
        for region, clusters in self.region_cluster.items():
            self.regions_weights[region] = 0
            for index in clusters:
                self.regions_weights[region] += self.clusters_weights[index]

############### generate edges ################
    def construct_edges(self):
        self.graph_edges = {}
        for source in self.counter.keys():
            for target, count in self.counter[source].items():
                self.add_to_graph(source, target, count)

        minus_1_id = self.name2index['-1']
        self.graph_edges[minus_1_id] = {}
        self.graph_edges[minus_1_id][minus_1_id] = 1

    def add_to_graph(self, source, target, count):
        source_id = self.name2index[source]
        target_id = self.get_target_id(source, target)
        self.initial_dict(self.graph_edges, source_id, {})
        self.initial_dict(self.graph_edges[source_id], target_id, 0)
        self.graph_edges[source_id][target_id] += count

    def get_target_id(self, source, target):
        source_list = source.strip().split('.')
        target_name = ""

        for b_id in range(1, len(source_list)):
            target_name += source_list[b_id] + "."
        target_name += target

        if target == '-1':
            target_id = self.name2index['-1']
        else:
            target_id = self.name2index[target_name]

        return target_id

############### display message ################
    def display(self, message):
        if self.verbose:
            print(message)

    def show_difference(self, title, save_path=None, critical_point=None):
        bin_inter = [i * 0.005 for i in range(32)]
        plt.figure()
        bins = plt.hist(self.global_difference, bins=bin_inter, density=True)
        bin_inter = [i * 0.02 for i in range(8)]
        xmin, xmax, ymin, ymax = plt.axis()
        if critical_point is not None:
            plt.vlines(critical_point, ymin=ymin, ymax=ymax, colors='r')
        plt.xticks(bin_inter)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.title(title)
        if save_path is None:
            #plt.show()
            pass
        else:
            plt.savefig(fname=(save_path + ".svg"), format='svg')

        return self.global_difference, bins
############### file operation ################
    def read_raw_data(self, data_path, InputFileDeliminator=' '):
        self.display("Load raw data message")

        RawTrajectories = []
        with open(data_path) as f:
            for line in f:
                fields = line.strip().split(InputFileDeliminator)
                ship = fields[0]

                #head = ['-1' for _ in range(1, self.order)]
                head = []

                movements = fields[1:]
                movements.append('-1')
                #movements = [key for key, grp in itertools.groupby(movements)]
                head.extend(movements)

                RawTrajectories.append(head)

        return RawTrajectories

############### generate other matrix ################
    def generate_trainsition_map(self, block_map):
        block_size = len(self.block2index)
        node_size = len(self.graph_nodes)
        map_matrix = np.zeros((node_size, block_size), dtype=np.float32)

        for id, nodes in enumerate(self.graph_nodes):
            block = self.last_element(nodes[0]) + "|"
            #block_index = self.block2index[block]
            block_index = block_map[block]
            map_matrix[id][block_index] = 1

        return map_matrix

    def generate_status_node_transition(self, status_map):
        status_size = len(status_map)
        node_size = len(self.graph_nodes)
        tran_matrix = np.zeros((status_size, node_size), dtype=np.float32)
        mask_matrix = np.zeros((status_size, node_size), dtype=np.float32)
        #count_matrix = np.zeros((status_size, node_size), dtype=np.float32)

        for source in self.distribution.keys():
            source_name = self.sequence2node(source)
            source_id = status_map[source_name]
            for target, pro in self.distribution[source].items():
                target_id = self.get_target_id(source, target)
                tran_matrix[source_id][target_id] = pro
                mask_matrix[source_id][target_id] = 1
                #count_matrix[self.status2index[source]][target_id] = self.counter[source][target]

        tran_matrix[status_size-1][node_size-1] = 1.0
        mask_matrix[status_size-1][node_size-1] = 1.0

        check_unique_axis(tran_matrix, axis=1)

        return tran_matrix, mask_matrix, #count_matrix[:-1, :]

    def get_status_node_map(self, status_map):
        status_size = len(status_map)
        node_size = len(self.graph_nodes)
        map_matrix = np.zeros((status_size, node_size), dtype=np.float32)
        for name, n_id in self.name2index.items():
            s_name = self.sequence2node(name)
            s_id = status_map[s_name]
            map_matrix[s_id][n_id] = 1
        return map_matrix

    def get_node_to_node_mask(self):
        node_size = len(self.graph_nodes)
        mask = np.zeros((node_size, node_size), dtype=np.float32)
        for source_id in self.graph_edges.keys():
            for target_id, _ in self.graph_edges[source_id].items():
                mask[source_id][target_id] = 1
        return mask


############### auxiliary function ################
    def initial_dict( self, in_dict, key, initial_value ) :
        if key not in in_dict:
            in_dict[key] = initial_value

    def increase_dict(self, in_dict, source, target, value=None):
        if source not in in_dict:
            in_dict[source] = {}
        if target not in in_dict[source]:
            in_dict[source][target] = 0

        if value is None:
            in_dict[source][target] += 1
        else:
            in_dict[source][target] = value

    def all_but_last_element(self, sequence):
        return_str = sequence[0]

        for s in sequence[1:-1]:
            return_str += ('.' + s)

        return return_str

    def check_for_mini_sup(self):
        for source in self.counter.keys():
            del_tar = []
            for target, count in self.counter[source].items():
                if count < self.mini_sup:
                    del_tar.append(target)
            assert len(del_tar) == 0
            for item in del_tar:
                self.counter[source].pop(item)

    def last_element(self, line):
        line_list = line.strip().split('.')
        return line_list[-1]


    def KLD_high_dimen(self, a, b):
        b = np.clip(b, 1e-07, 1)
        a_c = np.clip(a, 1e-07, 1)
        tmp = a_c / b
        tmp = np.log(tmp)
        tmp = tmp * a
        return np.sum(tmp, axis=1, keepdims=True)

    def KLD(self, a, b):
        b = np.clip(b, 1e-07, 1)
        a_c = np.clip(a, 1e-07, 1)
        tmp = a_c / b
        tmp = np.log(tmp)
        tmp = tmp * a
        return np.sum(tmp)

    def cosine_dist(self, a, b):
        return (1 - distance.cosine(a, b))

