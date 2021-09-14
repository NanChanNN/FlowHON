import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from nn.dataSetLoader import HoNLoader
from nn.dataSetLoader import HoNCluLoader


class autoTest:
    def __init__(self, name, Docu, train_file_seman, train_file_clu, valid_file_semantic, valid_file_clu, clusterSize=1000, step=8, alongTime=True, cluster_file_path=None, max_order=3):
        self.flowName = name
        self.step = step
        self.FoN = HoNLoader(name, 'FoN', Docu, train_file_seman, valid_file_semantic, clusterSize=clusterSize, max_order=max_order)
        #self.FoN.run()
        self.Ref = HoNCluLoader(name, 'Ref', Docu, train_file_clu, valid_file_clu, clusterSize=clusterSize, max_order=max_order)
        #self.Ref.run()
        self.Semantic = HoNLoader(name, 'Semantic', Docu, train_file_seman, valid_file_semantic, clusterSize=clusterSize, max_order=max_order)
        #self.Semantic.run()
        self.Clustering = HoNCluLoader(name, '', Docu, train_file_clu, valid_file_clu, clusterSize=clusterSize, file_path = cluster_file_path, max_order=max_order)
        #self.Clustering.run()

        self.alongTime = alongTime
        self.initialResult = {}
        self.result = {}

        self.randomWalkTestFoN()
        self.randomWalkTestSemantic()
        self.randomWalkTestRef()
        self.randomWalkTestClu()

        self.networkSize = {}
        self.lColor = {}
        self.lMarker = {}
        self.specifyColor()
        self.get_network_size()

    def get_network_size(self):
        self.networkSize["FoN"] = self.FoN.vertexNumber
        self.networkSize["Semantic"] = self.Semantic.vertexNumber
        self.networkSize["Clu"] = self.Clustering.vertexNumber
        self.networkSize["Ref"] = self.Ref.vertexNumber

    def randomWalkTestFoN(self, matrix = None, manipulate = None):
        if matrix is None:
            self.result['FoN'] = self.FoN.randomWalkTest(step=self.step, alongTime = self.alongTime)
            self.initialResult['FoN'] = self.result['FoN']
        else:
            mask = self.FoN.getMask()
            transM = self.manipulateMarix(matrix, manipulate, mask)
            self.result['FoN'] = self.FoN.randomWalkTest(proMat=transM, step=self.step, alongTime = self.alongTime)
        
        return np.mean(self.result['FoN'])

    def randomWalkTestSemantic(self, matrix = None, manipulate = None):
        if matrix is None:
            self.result['Semantic'] = self.Semantic.randomWalkTest(step=self.step, alongTime = self.alongTime)
            self.initialResult['Semantic'] = self.result['Semantic']
        else:
            mask = self.Semantic.getMask()
            transM = self.manipulateMarix(matrix, manipulate, mask)
            self.result['Semantic'] = self.Semantic.randomWalkTest(proMat=transM, step=self.step, alongTime = self.alongTime)
        
        return np.mean(self.result['Semantic'])

    def randomWalkTestRef(self, matrix = None, manipulate = None):
        if matrix is None:
            self.result['Ref'] = self.Ref.randomWalkTest(step=self.step, alongTime = self.alongTime)
            self.initialResult['Ref'] = self.result['Ref']
        else:
            mask = self.Ref.getMask()
            transM = self.manipulateMarix(matrix, manipulate, mask)
            self.result['Ref'] = self.Ref.randomWalkTest(proMat=transM, step=self.step, alongTime = self.alongTime)
        
        return np.mean(self.result['Ref'])
        
    def randomWalkTestClu(self, matrix = None, manipulate = None):
        if matrix is None:
            self.result['Clu'] = self.Clustering.randomWalkTest(step=self.step, alongTime = self.alongTime)
            self.initialResult['Clu'] = self.result['Clu']
        else:
            mask = self.Clustering.getMask()
            transM = self.manipulateMarix(matrix, manipulate, mask)
            self.result['Clu'] = self.Clustering.randomWalkTest(proMat=transM, step=self.step, alongTime = self.alongTime)
        
        return np.mean(self.result['Clu'])

    def randomWalkTestClu_model(self, in_model, name='Ref'):
        self.result[name] = self.Ref.randomWalkTest_model(model=in_model, step=self.step, alongTime=self.alongTime)

        return np.mean(self.result['Clu'])

    def showTestResult(self, initialRes = [], manipulation = "", save_path = None):
        x = np.arange(1, self.step+1, dtype=np.int32)
        plt.cla()

        for kind in self.result.keys():
            plt.plot(x, self.result[kind], color=self.lColor[kind], marker=self.lMarker[kind], ls='-', label=kind + " " + str(self.networkSize[kind]))

        for kind in initialRes:
            tempColor = list(self.lColor[kind])
            tempColor[-1] = 0.2
            plt.plot(x, self.initialResult[kind], color=tuple(tempColor), marker=self.lMarker[kind], ls='-', label="previous" + kind)

        print(self.initialResult['FoN'])

        title = self.flowName + " " + manipulation
        plt.suptitle(title, fontsize=16)

        plt.xticks(x)
        plt.ylabel('KLD')
        plt.xlabel('step')
        plt.legend()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(fname=(save_path+".svg"), format='svg')

    def showTestResult_v2(self, initialRes = [], manipulation = "", save_path = None):
        kind_name_map = {'FoN':'FoN', 'Semantic':'semantic', 'Ref':'ref'}
        kind_name_map_plus = {'FoN': 'FoN+', 'Semantic':'semantic+', 'Ref': 'ref+', 'Clu':'ours'}
        flow_name_map={'random-5cpR':'five critical points', 'abc':'ABC', 'bernard':'Bénard','crayfish':'crayfish',
                       'computer_room':'computer room','cylinder':'square cylinder', 'electro3D':'electron',
                       'plume':'solar plume', 'lifted_sub':'combustion', 'tornado':'tornado',
                       'two_swirl':'two swirls', 'hurricane_downsample':'hurricane'}

        x = np.arange(1, self.step + 1, dtype=np.int32)
        plt.cla()

        for kind in self.result.keys():
            plt.plot(x, self.result[kind], color=self.lColor[kind], marker=self.lMarker[kind], ls='-',
                     label=kind_name_map_plus[kind])

        for kind in initialRes:
            tempColor = list(self.lColor[kind])
            tempColor[-1] = 0.2
            plt.plot(x, self.initialResult[kind], color=tuple(tempColor), marker=self.lMarker[kind], ls='-',
                     label=kind_name_map[kind])

        title = flow_name_map[self.flowName] #+ " " + manipulation
        plt.suptitle(title, fontsize=16)

        plt.xticks(x)
        plt.ylabel('KLD')
        plt.xlabel('step')
        plt.legend()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(fname=(save_path + ".svg"), format='svg')

    def manipulateMarix(self, matrix, manipulate, mask):
        proMatrix = None
        if (manipulate=="Division"):
            maskM = matrix * mask
            absoluteW = np.abs(maskM)
            proMatrix = absoluteW / np.sum(absoluteW, 1, keepdims=True)

        elif (manipulate=="Softmax"):
            expW = np.exp(matrix) * mask
            proMatrix = expW / np.sum(expW, 1, keepdims=True)

        elif (manipulate=="Linear"):
            maskM = matrix * mask
            maskM = np.clip(maskM, a_min=0, a_max=None)
            proMatrix = maskM / np.sum(maskM, 1, keepdims=True)

        return proMatrix

    def specifyColor(self):
        self.lColor['FoN'] = tuple([0.122, 0.467, 0.706, 1.0])
        self.lColor['Ref'] = tuple([1.0, 0.498, 0.055, 1.0])
        self.lColor['Semantic'] = tuple([0.173, 0.627, 0.173, 1.0])
        self.lColor['Clu'] = tuple([0.839, 0.153, 0.157, 1.0])
        self.lMarker['FoN'] = 'o'
        self.lMarker['Ref'] = '^'
        self.lMarker['Semantic'] = 's'
        self.lMarker['Clu'] = '*'

    def store_matrix_weights(self, matrix, manipulate, mask, graph, output_file):
        transM = self.manipulateMarix(matrix, manipulate, mask)
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for source in graph.keys():
                for target in graph[source].keys():
                    writer.writerow([source, target, transM[source][target]])

    def showTestResult_v3(self, initialRes = [], manipulation = "", save_path = None):
        kind_name_map = {'FoN':'FoN', 'Semantic':'semantic', 'Ref':'ref'}
        kind_name_map_plus = {'FoN': 'FoN+', 'Semantic':'semantic+', 'Ref': 'ref+', 'Clu':'ours'}
        flow_name_map={'random-5cpR':'five critical points', 'abc':'ABC', 'bernard':'Bénard','crayfish':'crayfish',
                       'computer_room':'computer room','cylinder':'square cylinder', 'electro3D':'electron',
                       'plume':'solar plume', 'lifted_sub':'combustion', 'tornado':'tornado',
                       'two_swirl':'two swirls', 'hurricane_downsample':'hurricane'}
        write_row = []

        x = np.arange(1, self.step + 1, dtype=np.int32)
        plt.cla()

        for kind in self.result.keys():
            plt.plot(x, self.result[kind], color=self.lColor[kind], marker=self.lMarker[kind], ls='-',
                     label=kind_name_map_plus[kind])
            tmp = [kind]
            tmp.extend(self.result[kind])
            write_row.append(tmp)

        for kind in initialRes:
            tempColor = list(self.lColor[kind])
            tempColor[-1] = 0.2
            plt.plot(x, self.initialResult[kind], color=tuple(tempColor), marker=self.lMarker[kind], ls='-',
                     label=kind_name_map[kind])
            init_name = "init_" + kind
            tmp = [init_name]
            tmp.extend(self.initialResult[kind])
            write_row.append(tmp)

        title = flow_name_map[self.flowName] #+ " " + manipulation
        plt.suptitle(title, fontsize=16)

        plt.xticks(x)
        plt.ylabel('KLD')
        plt.xlabel('step')
        plt.legend()
        
        output_file = save_path + ".csv"
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for line in write_row:
                writer.writerow(line)

        if save_path is None:
            plt.show()
        else:
            pass
            #plt.savefig(fname=(save_path + ".svg"), format='svg')