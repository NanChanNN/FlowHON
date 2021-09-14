from utilityMy import *
import loader

def getHoNNodeName(trajectory, i, nodeSet, maxOrder):
    currentRegion = trajectory[i]
    nodeName = currentRegion + "|"
    tempName = nodeName
    order = 1
    curr = i

    while curr > 0 and order <= maxOrder:
        curr -= 1
        order += 1

        if order == 2:
            tempName += trajectory[curr]
        else:
            tempName += "." + trajectory[curr]

        if tempName in nodeSet:
            nodeName = tempName

    return nodeName

def getCurrentName(node):
    spList = node.strip().split("|")
    return spList[0]

def getNodeWorkLoad(nodeSet, nodeTracingTime, TrainingTrajectory, maxOrder):
    workloadMap = {}
    returnWorkloadMapAver = {}
    returnWorkloadMapMean = {}
    for index in range(len(TrainingTrajectory)):
        trajectory = TrainingTrajectory[index][1][:-1]
        time = nodeTracingTime[index]

        for i in range(len(trajectory)):
            NodeName = getHoNNodeName(trajectory, i, nodeSet, maxOrder)
            currentName = getCurrentName(NodeName)
            timeTuple = time[i]
            if int(currentName) == timeTuple[0]:
                if NodeName not in workloadMap:
                    workloadMap[NodeName] = []
                workloadMap[NodeName].append(timeTuple[1])

    for key, value in workloadMap.items():
        returnWorkloadMapAver[key] = np.around(np.average(value) , 2)
        returnWorkloadMapMean[key] = np.around(np.median(value), 2)

    returnWorkloadMapAver['-1|'] = 0
    returnWorkloadMapMean['-1|'] = 0

    remainSet =  set(nodeSet) - set(returnWorkloadMapAver.keys())

    for remain in remainSet:
        tempListAver = []
        tempListMean = []

        for key, value in returnWorkloadMapAver.items():
            if key.find(remain) != -1:
                tempListAver.append(value)
                tempListMean.append(returnWorkloadMapMean[key])
        
        returnWorkloadMapAver[remain] = np.around(np.average(tempListAver), 2)
        returnWorkloadMapMean[remain] = np.around(np.median(tempListMean), 2)
    
    return returnWorkloadMapAver, returnWorkloadMapMean



def writeGraphInfo(maxOrder, miniSupp, flowName, Docu, nodeTracingTime, TrainingTrajectory):

    fileName = Docu + "/Data/"+flowName + "/network-" + flowName + "-" + str(maxOrder) + "-" + str(miniSupp) + "_END.csv"

    directedEdgeDict = {}
    node2index = {}
    FoNNodeSet = []

    with open(fileName, 'r') as f:
        edgeLines = f.readlines()

        edgeNum = len(edgeLines)
        print("Total Lines:", edgeNum)

    for edge in edgeLines:
        edge = edge.strip("\n")
        edge = edge.strip(" ")
        edgeArr = edge.strip().split(",")

        if(isFoNNode(edgeArr[0])):
            if edgeArr[0] not in node2index:
                FoNNodeSet.append(edgeArr[0])
                node2index[edgeArr[0]] = len(node2index)

        if(isFoNNode(edgeArr[1])):
            if edgeArr[1] not in node2index:
                FoNNodeSet.append(edgeArr[1])
                node2index[edgeArr[1]] = len(node2index)

        if edgeArr[0] not in directedEdgeDict:
            directedEdgeDict[edgeArr[0]] = {}

        if(edgeArr[1]=="-1|"):
            directedEdgeDict[edgeArr[0]][edgeArr[1]] = math.ceil(int(edgeArr[2]) * 1.0)
        else:
            directedEdgeDict[edgeArr[0]][edgeArr[1]] = int(edgeArr[2])
    
    for edge in edgeLines:
        edge = edge.strip("\n")
        edge = edge.strip(" ")
        edgeArr = edge.strip().split(",")

        if edgeArr[0] not in node2index:
            node2index[edgeArr[0]] = len(node2index)

        if edgeArr[1] not in node2index:
            node2index[edgeArr[1]] = len(node2index)

    workLoadMapAver, workLoadMean = getNodeWorkLoad(node2index.keys(), nodeTracingTime, TrainingTrajectory, maxOrder)
    workLoadMapAverRegion, workLoadMapMeanRegion = getNodeWorkLoad(list(FoNNodeSet), nodeTracingTime, TrainingTrajectory, 1)

    loader.loadGraphToFile(flowName, FoNNodeSet, node2index, directedEdgeDict, maxOrder, Docu, workLoadMapAver, workLoadMean, workLoadMapAverRegion, workLoadMapMeanRegion)