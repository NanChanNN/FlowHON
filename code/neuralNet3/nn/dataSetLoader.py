import itertools
import numpy as np
import utility
import csv

class HoNLoader:

    def __init__(self, flowName, status, docu, train_file, valid_file, clusterSize=1000, lengthPerClu=10, splitRate=0.2, setted_model=None, max_order = 3, graph_file=None):
        self.clusterSize = clusterSize
        self.lengthPerClu = lengthPerClu
        self.splitRate = splitRate
        self.max_order = max_order
        self.document_path = docu
        self.stlFile = train_file
        self.validFile = valid_file

        self.flowName = flowName
        self.status = status
        self.graphFile = graph_file
        self.transitionMatrix = None
        self.mask = None
        self.mapMatrix = None
        self.data_X = None
        self.data_Y = None
        self.once = False
        self.rerun = False
        if graph_file is None:
            self.getFileName()
        else:
            self.order = 1
        self.run(hon_network=setted_model)

    def run(self, hon_network=None):
        if hon_network is None:
            self.constructHoN()
        else:
            self.constructHoN_from_HoN(hon_network)
        self.getMapNode2Node()
        self.getDataFrame()
        self.getXandY()
        self.getTransitionProb()
        self.getValidationData()

    def getFileName(self):
        documentPath = self.document_path + "/Data/" + self.flowName + "/"
        #self.stlFile = documentPath + self.flowName + '-DataSequen.csv'
        #self.validFile =  "../../Data/" + self.flowName + "/" + self.flowName + "-DataSequenVal.csv"

        assert self.status in ["FoN", "Semantic"]
        if self.status == "FoN":
            post = ''
            if self.max_order!=3:
                post = "-o" + str(self.max_order)
            post = ''
            self.order = 1
            self.graphFile = documentPath + self.flowName + "1-graph" + ".csv"

        elif self.status == "Semantic":
            self.order = self.max_order
            self.graphFile = documentPath + self.flowName + str(self.order) + "-graph.csv"

    def constructHoN_from_HoN(self, hon_network):
        nodeToIndex = {}  # convert node name to index
        regionIndex = {}  # convert region name to index
        vertexInfo = {}  # convert vertex index to tuple (weight, regionIndex)
        graph = {}

        regionNumber = hon_network.block_num
        vertexNumber = hon_network.node_num

        for block, index in hon_network.block2index.items():
            regionName = block + "|"
            regionIndex[regionName] = index
        for id, nodes in enumerate(hon_network.graph_nodes):
            block = hon_network.last_element(nodes[0])
            regionName = block + "|"
            vertexInfo[id] = tuple([hon_network.clusters_weights[id], regionIndex[regionName]])
            for n in nodes:
                nodeToIndex[hon_network.sequence2node(n)] = id

        for source_id in hon_network.graph_edges.keys():
            graph[source_id] = {}
            for target_id, value in hon_network.graph_edges[source_id].items():
                graph[source_id][target_id] = value

        self.nodeToIndex = nodeToIndex
        self.regionIndex = regionIndex
        self.vertexInfo = vertexInfo
        self.graph = graph
        self.regionNumber = regionNumber
        self.vertexNumber = vertexNumber

    def constructHoN(self):
        nodeToIndex = {} # convert node name to index
        regionIndex = {} # convert region name to index
        vertexInfo = {} # convert vertex index to tuple (weight, regionIndex)
        graph = {}

        with open(self.graphFile) as f:
            fList = list(f)

            fields = fList[0].strip().split(',')
            regionNumber = int(fields[0])
            vertexNumber = int(fields[1])

            for regionCounter in range(regionNumber):
                fields = fList[1 + regionCounter].strip().split(',')
                regionName = fields[0] + "|"
                regionIndex[regionName] = int(fields[1])

            for vertexCounter in range(vertexNumber):
                fields = fList[1 + regionNumber + vertexCounter].strip().split(',')
                assert len(fields) == 3
                regionName = fields[0] + "|"
                vertexInfo[vertexCounter] = tuple([int(fields[-1]), regionIndex[regionName]])
                for k in range(1, len(fields)-1):
                    nodeToIndex[fields[k]] = vertexCounter
                
            for edgeCounter in range(1 + regionNumber + vertexNumber, len(fList)):
                fields = fList[edgeCounter].strip().split(',')
                if int(fields[0]) not in graph:
                    graph[int(fields[0])] = {}
                graph[int(fields[0])][int(fields[1])] = int(fields[2])

        self.nodeToIndex = nodeToIndex
        self.regionIndex = regionIndex
        self.vertexInfo = vertexInfo
        self.graph = graph
        self.regionNumber = regionNumber
        self.vertexNumber = vertexNumber

    def getTransitionProb(self):
        if self.transitionMatrix is None or self.rerun:
            self.transitionMatrix = np.zeros((len(self.vertexInfo), len(self.vertexInfo)), dtype=np.float32)
            self.mask = np.zeros((len(self.vertexInfo), len(self.vertexInfo)), dtype=np.float32)
            for source in self.graph.keys():
                total = sum(self.graph[source].values())
                for target, value in self.graph[source].items():
                    self.transitionMatrix[source][target] = np.float32(value/total)
                    self.mask[source][target] = 1

        return self.transitionMatrix

    def getMask(self):
        if self.mask is None or self.rerun:
            self.getTransitionProb()
        
        return  self.mask

    def getMapNode2Node(self):
        if not(self.mapMatrix is None) and  not (self.rerun):
            return self.mapMatrix

        self.mapMatrix = np.zeros((len(self.vertexInfo), len(self.regionIndex)), dtype=np.float32)
        for key, value in self.vertexInfo.items():
            self.mapMatrix[key][value[1]] = 1

        return self.mapMatrix

    def getDataFrame(self):
        self.dataFrame = []
        with open(self.stlFile) as f:
            for line in f:
                fields = line.strip().split(' ')
                movements = fields[1:]
                movements = [key for key,grp in itertools.groupby(movements)]
                self.dataFrame.append(movements)

    def getXandY(self):
        self.numOfcluster = int(len(self.dataFrame) //self.clusterSize)
        self.X = []
        self.Y = []
        self.Y_2 = []

        for cId in range(self.numOfcluster):
            for step in range(self.lengthPerClu):
                x = self.getInput(cId, step)
                y = self.getOuput(cId, step)
                y_2 = self.getInput(cId, step+1)

                self.X.append(x)
                self.Y.append(y)
                self.Y_2.append(y_2)

    def getInput(self, cId, step):
        begin = cId * self.clusterSize
        end = (cId+1) * self.clusterSize
        columnID = step
        itemTensor = np.zeros(len(self.vertexInfo))

        for item in self.dataFrame[begin:end]:
            nodeName = self.getNodeName(item, columnID, self.order)
            itemTensor[self.nodeToIndex[nodeName]] += 1

        return itemTensor

    def getOuput(self, cId, step):
        begin = cId * self.clusterSize
        end = (cId + 1) * self.clusterSize
        columnID = step + 1
        itemTensor = np.zeros(len(self.regionIndex))

        for item in self.dataFrame[begin:end]:
            nodeName = self.getNodeName(item, columnID, 1)
            itemTensor[self.regionIndex[nodeName]] += 1
        
        return itemTensor

    def display_once(self, message):
        if not self.once:
            print(message)
        self.once = True

    def check_region(self, region):
        ret_region = int(region)
        tempName = str(ret_region) + "|"
        count = 1
        while tempName not in self.regionIndex:
            self.display_once("**********WARNING: Result might not be accurate!**********")
            if tempName not in self.regionIndex:
                ret_region = int(region) + count
                tempName = str(ret_region) + "|"
            if tempName not in self.regionIndex:
                ret_region = int(region) - count
                tempName = str(ret_region) + "|"
            count = count + 1

        return str(ret_region)

    def getNodeName(self, movement, index, order):
        if(index >= len(movement)):
            name = "-1|"
        else:
            #region = self.check_region(movement[index])
            region = movement[index]
            tempName = region + "|"

            name = tempName
            for i in range(1, order):
                
                if index < i:
                    break

                if i == 1 :
                    tempName += movement[index - i]
                else:
                    tempName += "." + movement[index - i]

                if tempName in self.nodeToIndex:
                    name = tempName

        return name

    def splitTrainAndValid(self, splitRate = 0.2):
        dataTrainSize = int((1-splitRate) * len(self.X))
        return np.array(self.X[0:dataTrainSize]), np.array(self.Y[0:dataTrainSize]), np.array(self.X[dataTrainSize:]), np.array(self.Y[dataTrainSize:])

    def splitTrainAndValid_V2(self, splitRate = 0.0):
        dataTrainSize = int((1-splitRate) * len(self.X))
        return np.array(self.X[0:dataTrainSize], dtype=np.float32), np.array(self.Y[0:dataTrainSize], dtype=np.float32), \
               np.array(self.Y_2[0:dataTrainSize], dtype=np.float32), np.array(self.X[dataTrainSize:], dtype=np.float32), \
               np.array(self.Y[dataTrainSize:], dtype=np.float32), np.array(self.Y_2[dataTrainSize:], dtype=np.float32)

    def randomWalkTest(self, proMat = None, step=8, alongTime=True):
        if proMat is None:
            proMat = self.transitionMatrix
        
        begin = 0
        sampleSize = self.sample_size

        initialVec = self.valX[begin]
        initialVec = initialVec.reshape((1, self.vertexNumber))

        intermediateResult = []

        for s in range(step):

            if alongTime:
                initialVec = np.matmul(initialVec, proMat)
            else:
                initialVec = self.valX[begin+s]
                initialVec = initialVec.reshape((1,self.vertexNumber))
                initialVec = np.matmul(initialVec, proMat)

            y_pred = np.matmul(initialVec, self.mapMatrix)

            y_true = self.valY[begin+s]
            y_true = y_true.reshape((1, self.regionNumber))

            intermediateResult.append(utility.KLD(y_pred, y_true, sampleSize))

        return intermediateResult

    def getValidationData(self):
        dataFrame = []
        self.valX = []
        self.valY = []
        with open(self.validFile) as f:
            for line in f:
                fields = line.strip().split(' ')
                movements = fields[1:]
                movements = [key for key,grp in itertools.groupby(movements)]
                dataFrame.append(movements)

            self.sample_size = len(dataFrame)

            for step in range(self.lengthPerClu):
                columnID = step
                itemTensor = np.zeros(len(self.vertexInfo))
                itemTensor2 = np.zeros(len(self.regionIndex))
                for item in dataFrame:
                    nodeName = self.getNodeName(item, columnID, self.order)
                    itemTensor[self.nodeToIndex[nodeName]] += 1

                    regionName = self.getNodeName(item, columnID+1, 1)
                    itemTensor2[self.regionIndex[regionName]] += 1

                self.valX.append(itemTensor)
                self.valY.append(itemTensor2)

class HoNCluLoader:

    def __init__(self, flowName, status, Docu, train_file, valid_file, clusterSize=1000, lengthPerClu=10, setted_model = None, file_path = None, max_order=3):

        self.clusterSize = clusterSize
        self.lengthPerClu = lengthPerClu

        self.document_path = Docu
        self.validFile = valid_file
        self.stlFile = train_file
        self.max_order = max_order

        self.flowName = flowName
        self.status = status
        self.graphFile = None
        self.transitionMatrix = None
        self.mask = None
        self.mapMatrix = None
        self.data_X = None
        self.data_Y = None
        self.once = False
        self.rerun = False

        self.intial_count = None

        if file_path is None:
            self.getFileName()
        else:
            self.order = self.max_order
            self.graphFile = file_path

        self.run(hon_network=setted_model)

    def run(self, hon_network=None):
        if hon_network is None:
            self.constructHoN()
            self.get_initial_count() # approximate
        else:
            self.constructHoN_from_HON(hon_network)
        self.getMapNode2Node()
        self.getDataFrame()
        self.getXandY()
        self.getTransitionProb()
        self.getValidationData()

    def getFileName(self):
        documentPath = self.document_path + "/Data/" + self.flowName + "/"
        #documentPath = "../../Data/" + self.flowName + "/"
        #self.stlFile = documentPath + self.flowName + '-DataSequen.csv'
        #self.validFile =  "../../Data/" + self.flowName + "/" + self.flowName + "-DataSequenVal.csv"

        if self.status == "Ref":
            self.order = self.max_order
            self.graphFile = documentPath + self.flowName + str(self.order) + "-graphRef.csv"
        else:
            self.order = self.max_order
            self.graphFile = documentPath + self.flowName + str(self.order) + "-graphClu.csv"

    def constructHoN_from_HON(self, hon_network):
        nodeToIndex = {}  # convert node name to index
        regionIndex = {}  # convert region name to index
        vertexInfo = {}  # convert vertex index to tuple (weight, regionIndex)
        graph = {}
        vertex_status = []
        regionsVertex = {}
        indexRegion = {}

        regionNumber = hon_network.block_num
        vertexNumber = hon_network.node_num

        for block, index in hon_network.block2index.items():
            regionName = block + "|"
            regionIndex[regionName] = index
            indexRegion[index] = regionName
        for id, nodes in enumerate(hon_network.graph_nodes):
            block = hon_network.last_element(nodes[0])
            regionName = block + "|"
            vertexInfo[id] = tuple([hon_network.clusters_weights[id], regionIndex[regionName]])

            if regionName not in regionsVertex:
                regionsVertex[regionName] = []
            regionsVertex[regionName].append(tuple([id, hon_network.clusters_weights[id]]))

            tmp_list = []
            for n in nodes:
                field_k = hon_network.sequence2node(n)
                nodeToIndex[field_k] = id
                tmp_list.append(field_k)
            vertex_status.append(tmp_list)

        for source_id in hon_network.graph_edges.keys():
            graph[source_id] = {}
            for target_id, value in hon_network.graph_edges[source_id].items():
                graph[source_id][target_id] = value

        self.intial_count = {}
        for id, count in enumerate(hon_network.record_list):
            self.intial_count[id] = int(count)

        self.indexRegion = indexRegion
        self.nodeToIndex = nodeToIndex
        self.regionIndex = regionIndex
        self.vertexInfo = vertexInfo
        self.regionsVertex = regionsVertex
        self.graph = graph
        self.regionNumber = regionNumber
        self.vertexNumber = vertexNumber
        self.vertex_status = vertex_status

    def constructHoN(self):
        nodeToIndex = {} # convert node name to index
        regionIndex = {} # convert region name to index
        vertexInfo = {} # convert vertex index to tuple (weight, regionIndex)
        vertex_status = []
        regionsVertex = {}
        indexRegion = {}
        graph = {}

        with open(self.graphFile) as f:
            fList = list(f)

            fields = fList[0].strip().split(',')
            regionNumber = int(fields[0])
            vertexNumber = int(fields[1])

            for regionCounter in range(regionNumber):
                fields = fList[1 + regionCounter].strip().split(',')
                regionName = fields[0] + "|"
                regionIndex[regionName] = int(fields[1])
                indexRegion[int(fields[1])] = regionName

            for vertexCounter in range(vertexNumber-1):
                fields = fList[1 + regionNumber + vertexCounter].strip().split(',')
                regionName = fields[0] + "|"
                vertexInfo[vertexCounter] = tuple([float(fields[-1]), regionIndex[regionName]])

                if regionName not in regionsVertex:
                    regionsVertex[regionName] = []
                regionsVertex[regionName].append(tuple([vertexCounter, int(float(fields[-1]))]))

                tmp_list = []
                for k in range(1, len(fields)-1):
                    nodeToIndex[fields[k]] = vertexCounter
                    tmp_list.append(fields[k])
                vertex_status.append(tmp_list)

            fields = fList[1 + regionNumber + vertexNumber - 1].strip().split(',')
            vertexInfo[vertexNumber-1] = tuple([float(fields[-1]), regionIndex["-1|"]])
            nodeToIndex["-1|"] = vertexNumber - 1
            vertex_status.append(["-1|"])
            regionsVertex["-1|"] = []
            regionsVertex["-1|"].append(tuple([vertexNumber-1, 0]))
                
            for edgeCounter in range(1 + regionNumber + vertexNumber, len(fList)):
                fields = fList[edgeCounter].strip().split(',')
                if int(fields[0]) not in graph:
                    graph[int(fields[0])] = {}
                graph[int(fields[0])][int(fields[1])] = float(fields[2])

        self.indexRegion = indexRegion
        self.nodeToIndex = nodeToIndex
        self.regionIndex = regionIndex
        self.vertexInfo = vertexInfo
        self.regionsVertex = regionsVertex
        self.graph = graph
        self.regionNumber = regionNumber
        self.vertexNumber = vertexNumber
        self.vertex_status = vertex_status

    def getTransitionProb(self):
        if self.transitionMatrix is None or self.rerun:
            self.transitionMatrix = np.zeros((len(self.vertexInfo), len(self.vertexInfo)), dtype=np.float32)
            self.transitionMatrix_count = np.zeros((len(self.vertexInfo), len(self.vertexInfo)), dtype=np.float32)
            self.mask = np.zeros((len(self.vertexInfo), len(self.vertexInfo)), dtype=np.float32)
            for source in self.graph.keys():
                total = sum(self.graph[source].values())
                for target, value in self.graph[source].items():
                    self.transitionMatrix[source][target] = np.float32(value/total)
                    self.transitionMatrix_count[source][target] = np.float(value)
                    self.mask[source][target] = 1

        return self.transitionMatrix

    def getMask(self):
        if self.mask is None or self.rerun:
            self.getTransitionProb()
        
        return  self.mask

    def getMapNode2Node(self):
        if not (self.mapMatrix is None) and not (self.rerun):
            return self.mapMatrix

        self.mapMatrix = np.zeros((len(self.vertexInfo), len(self.regionIndex)), dtype=np.float32)
        for key, value in self.vertexInfo.items():
            self.mapMatrix[key][value[1]] = 1

        return self.mapMatrix

    def get_initial_count(self):
        intiial_count_file = self.graphFile[:-4] + '-intial.csv'
        self.intial_count = {}
        with open(intiial_count_file) as f:
            for line in f:
                fields = line.strip().split(',')
                self.intial_count[int(fields[0])] = int(fields[1])


    def getDataFrame(self):
        self.dataFrame = []
        with open(self.stlFile) as f:
            for line in f:
                
                finalList = []
                for _ in range(1, self.order):
                    #finalList.append('-1')
                    pass

                fields = line.strip().split(' ')
                movements = fields[1:]
                #movements = [key for key,grp in itertools.groupby(movements)]

                finalList.extend(movements)

                self.dataFrame.append(finalList)

    def getXandY(self):
        self.numOfcluster = int(len(self.dataFrame) //self.clusterSize)

        index = list(range(len(self.dataFrame)))
        #import random
        #random.seed(0)
        #random.shuffle(index)
        self.index = index

        self.X = []
        self.Y = []
        self.Y_2 = []

        for cId in range(self.numOfcluster):
            for step in range(self.lengthPerClu):
                x = self.getInput(cId, step)
                y = self.getOuput(cId, step)
                y_2 = self.getInput(cId, step + 1)

                self.X.append(x)
                self.Y.append(y)
                self.Y_2.append(y_2)

    def getInput(self, cId, step):
        begin = cId * self.clusterSize
        end = (cId+1) * self.clusterSize
        columnID = step + self.order - 1
        itemTensor = np.zeros(len(self.vertexInfo))

        #for item in self.dataFrame[self.index[begin:end]]:
        for id in self.index[begin:end]:
            item = self.dataFrame[id]
            nodeName = self.getNodeName(item, columnID, self.order)
            if nodeName in self.nodeToIndex:
                itemTensor[self.nodeToIndex[nodeName]] += 1
            else:
                distPair = self.getDistribution(nodeName)
                for pair in distPair:
                    itemTensor[pair[0]] += pair[1]

        return itemTensor

    def getOuput(self, cId, step):
        begin = cId * self.clusterSize
        end = (cId + 1) * self.clusterSize
        columnID = (step + 1) + self.order - 1
        itemTensor = np.zeros(len(self.regionIndex))

        for item in self.dataFrame[begin:end]:
            nodeName = self.getNodeName(item, columnID, 1)
            itemTensor[self.regionIndex[nodeName]] += 1
        
        return itemTensor

    def getNodeName(self, movement, index, order):
        order -= 1
        if(index >= len(movement)):
            name = "-1|"
        else:
            #name = movement[index] + "|"
            #region = self.check_region(name)
            region = movement[index]
            name = region + "|"

            for i in range(order):
                if i == 0:
                    name += movement[index - i -1]
                else:
                    name += "." + movement[index - i -1]
        return name

    def display_once(self, message):
        if not self.once:
            print(message)
        self.once = True

    def check_region(self, current_node):
        tempName = current_node
        region = current_node.strip().split('|')[0]
        ret_region = region
        count = 1
        while tempName not in self.regionIndex:
            self.display_once("**********WARNING: Result might not be accurate!**********")
            if tempName not in self.regionIndex:
                ret_region = int(region) + count
                tempName = str(ret_region) + "|"
            if tempName not in self.regionIndex:
                ret_region = int(region) - count
                tempName = str(ret_region) + "|"
            count = count + 1

        return str(ret_region)

    def getDistribution(self, nodeName):
        retArr = []
        totalWeight = 0

        currentNode = utility.getCurrentNode(nodeName)
        #currentNode = self.check_region(currentNode)
        #currentNode = currentNode + "|"
        vertexList = self.regionsVertex[currentNode]
        for vertex in vertexList:
            totalWeight += vertex[1]
        for vertex in vertexList:
            retArr.append(tuple([vertex[0], vertex[1]/totalWeight]))

        return retArr

    def splitTrainAndValid(self, splitRate = 0.0):
        dataTrainSize = int((1-splitRate) * len(self.X))
        return np.array(self.X[0:dataTrainSize]), np.array(self.Y[0:dataTrainSize]), np.array(self.X[dataTrainSize:]), np.array(self.Y[dataTrainSize:])

    def splitTrainAndValid_V2(self, splitRate = 0.0):
        dataTrainSize = int((1-splitRate) * len(self.X))
        return np.array(self.X[0:dataTrainSize], dtype=np.float32), np.array(self.Y[0:dataTrainSize], dtype=np.float32), \
               np.array(self.Y_2[0:dataTrainSize], dtype=np.float32), np.array(self.X[dataTrainSize:], dtype=np.float32), \
               np.array(self.Y[dataTrainSize:], dtype=np.float32), np.array(self.Y_2[dataTrainSize:], dtype=np.float32)

    def getValidationData(self):
        dataFrame = []
        self.valX = []
        self.valY = []

        with open(self.validFile) as f:
            for line in f:
                finalList = []
                for _ in range(1, self.order):
                    #finalList.append('-1')
                    pass

                fields = line.strip().split(' ')
                movements = fields[1:]
                #movements = [key for key, grp in itertools.groupby(movements)]
                finalList.extend(movements)

                dataFrame.append(finalList)

            self.sample_size = len(dataFrame)
            begin = 0

            itemTensor, itemTensor2 = self.get_initial_distri(dataFrame) # approximate
            self.valX.append(itemTensor) # approximate
            self.valY.append(itemTensor2) # approximate
            begin = 1 # approximate

            for step in range(begin, self.lengthPerClu):
                columnID = step + self.order - 1
                columnID = step # approximate
                itemTensor = np.zeros(len(self.vertexInfo))
                itemTensor2 = np.zeros(len(self.regionIndex))
                for id, item in enumerate(dataFrame):
                    nodeName = self.getNodeName(item, columnID, self.order)
                    if nodeName in self.nodeToIndex:
                        itemTensor[self.nodeToIndex[nodeName]] += 1
                    else:
                        distPair = self.getDistribution(nodeName)
                        for pair in distPair:
                            itemTensor[pair[0]] += pair[1]

                    regionName = self.getNodeName(item, columnID+1, 1)
                    itemTensor2[self.regionIndex[regionName]] += 1

                self.valX.append(itemTensor)
                self.valY.append(itemTensor2)

    def get_initial_distri(self, data_frame):
        itemTensor = np.zeros(len(self.vertexInfo))
        itemTensor2 = np.zeros(len(self.regionIndex))

        for id, item in enumerate(data_frame):
            block_name = item[0] + '|'
            distPair = self.get_initial_block_distri(block_name)
            for pair in distPair:
                itemTensor[pair[0]] += pair[1]

            regionName = self.getNodeName(item, 0 + 1, 1)
            itemTensor2[self.regionIndex[regionName]] += 1

        return itemTensor, itemTensor2

    def get_initial_block_distri(self, block_name):
        retArr = []
        vertexList = self.regionsVertex[block_name]
        totalWeight = 0
        totalOutWeight = 0
        for vertex in vertexList:
            if vertex[0] in self.intial_count:
                totalWeight += self.intial_count[vertex[0]]
            else:
                totalWeight += 0
            totalOutWeight += vertex[1]
        if totalWeight != 0:
            for vertex in vertexList:
                retArr.append(tuple([vertex[0], self.intial_count[vertex[0]] / totalWeight]))
        else:
            for vertex in vertexList:
                retArr.append(tuple([vertex[0], vertex[1] / totalOutWeight]))
        
        return retArr

    def construct_nodeDistri_mat(self, proMat = None):
        if proMat is None:
            proMat = self.transitionMatrix

        self.nodeDistri = {}

        for index in range(self.vertexNumber):
            record = {}
            nonZeroID = np.nonzero(proMat[index, :])[0]
            nonZeroID = nonZeroID.tolist()

            for id in nonZeroID:
                curNode = utility.getCurrentNodeIne(self.indexRegion[self.vertexInfo[id][1]])
                if curNode not in record:
                    record[curNode] = 0.0
                record[curNode] += proMat[index, id]

            for key in record.keys():
                record[key] = float(record[key] * self.vertexInfo[index][0])

            self.nodeDistri[index] = record

    def write_trained_status(self, proMat = None, mnip="Dvision"):
        self.construct_nodeDistri_mat(proMat)

        documentPath = "../../Data/" + self.flowName + "/"
        if mnip == "Division":
            outputFile = documentPath + self.flowName + "_trained_status_div.csv"
        else:
            outputFile = documentPath + self.flowName + "_trained_status_lin.csv"

        with open(outputFile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for node, index in self.nodeToIndex.items():
                if(node=="-1|"):
                    continue
                for key, value in self.nodeDistri[index].items():
                    row = [node, key, value]
                    writer.writerow(row)

    def randomWalkTest_model(self, model, step=8, alongTime=True):
        sampleSize = self.sample_size
        #print("sampleSIze:", sampleSize)
        begin = 0
        initialVec = self.valX[begin]
        initialVec = initialVec.reshape((1, self.vertexNumber))

        intermediateResult = []
        model.clean_inference()
        model.re_initial()
        model.check_condition()

        for s in range(step):
            if (alongTime):
                y_pred = model.inference(initialVec, alongTime=True)
            else:
                initialVec = self.valX[begin + s]
                initialVec = initialVec.reshape((1, self.vertexNumber))
                y_pred = model.inference(initialVec, alongTime=False)

            y_true = self.valY[begin + s]
            y_true = y_true.reshape((1, self.regionNumber))
            intermediateResult.append(utility.KLD(y_pred, y_true, sampleSize))

        return intermediateResult

    def randomWalkTest(self, proMat=None, step=8, alongTime=True):
        if proMat is None:
            proMat = self.transitionMatrix

        #sampleSize = 5000
        sampleSize = self.sample_size

        begin = 0
        initialVec = self.valX[begin]
        initialVec = initialVec.reshape((1, self.vertexNumber))

        intermediateResult = []

        for s in range(step):
            if (alongTime):
                initialVec = np.matmul(initialVec, proMat)
            else:
                initialVec = self.valX[begin + s]
                initialVec = initialVec.reshape((1, self.vertexNumber))
                initialVec = np.matmul(initialVec, proMat)

            y_pred = np.matmul(initialVec, self.mapMatrix)

            y_true = self.valY[begin + s]
            y_true = y_true.reshape((1, self.regionNumber))
            intermediateResult.append(utility.KLD(y_pred, y_true, sampleSize))

        return intermediateResult







