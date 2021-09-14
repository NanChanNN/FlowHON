import numpy as np
from scipy import special
import scipy.stats
import math
import csv

def decomposeNode(nodeName):
    tempArr = nodeName.split("|")
    returnArr = []
    
    returnArr.append(int(tempArr[0]))
    tempArr2 = tempArr[1].split(".")

    if tempArr2[0] == '':
        returnArr.append([])
    else:
        returnArr.append(list(map(int, tempArr2)))

    return returnArr

def getNodeOrder(nodeName):
    node = decomposeNode(nodeName)
    order = len(node[1]) + 1
    return order

def isFoNNode(nodeName):
    if getNodeOrder(nodeName) == 1:
        return True
    else:
        return False

def getCurrentNode(nodeName):
    node = decomposeNode(nodeName)
    return (str(node[0]) + "|")

def generateNodeMap(FoNNodeSet, HoNNodeSet):
    returnMap = {}
    for fNode in FoNNodeSet:
        returnMap[fNode] = len(returnMap)
    for hNode in HoNNodeSet:
        returnMap[hNode] = len(returnMap)

    #print(returnMap) 

    return returnMap

def constructTransMat(node2index, directedEdgeDict):
    matrixSize = len(node2index)
    transitionMat = np.zeros((matrixSize, matrixSize), dtype=np.float64)
    
    for source in directedEdgeDict:
        sourceIndex = node2index[source]
        edgeSum = np.float64(sum(directedEdgeDict[source].values()))
        for target in directedEdgeDict[source]:
            targetIndex = node2index[target]
            weight = directedEdgeDict[source][target]
            transitionMat[targetIndex][sourceIndex] = np.float64(weight) / edgeSum

    transitionMat[node2index["-1|"]][node2index["-1|"]] = 1.0
    return transitionMat

def neighborhood(node2index, num, r):
    for i in range(1,r+1):
        name = str(num+i) + "|"
        if name in node2index :
            return name
        name = str(num-i) + "|"
        if name in node2index :
            return name
    return "-1|"

def getNodeIndex(node2index, num):
    name = str(num) + "|"

    if(name not in node2index):
        name = neighborhood(node2index, num, 1)
    return node2index[name]

def produceDistribution(node2index, flowName, step, matrixSize):
    #matrixSize = len(node2index)
    resultDist = np.zeros((matrixSize, 1), dtype=np.float64, order='F')
    fileName = "../input_data/"+flowName +"/" + flowName + "-DataSequen_TESTING.csv"

    with open(fileName, 'r') as f:
        testLines = f.readlines()
        totalNum = np.float64(len(testLines))
        print("Total Lines:", totalNum)

    for line in testLines:
        line = line.strip("\n")
        line = line.strip(" ")
        testArr = line.split(",")

        if testArr[0] == '':
            testArr = testArr[1:]

        if len(testArr) <= step:
            index = getNodeIndex(node2index, -1)
        else:
            index = getNodeIndex(node2index, int(testArr[step]))

        resultDist[index] += 1
    
    return resultDist #/ totalNum


def KL_Divergence(trueD, testD):
    #arr = special.rel_entr(trueD, testD)
    size = trueD.shape[0]
    if size != testD.shape[0]:
        print("ERROR!")
        return 0
    sumR = 0.0 
    for i in range(size):
        if trueD[i]==0 or testD[i] == 0:
            continue
        else:
            sumR += (trueD[i]/10000) * (math.log(trueD[i]/testD[i]))

    return sumR

def jensen_shannon_distance(trueD, testD):
    p = np.array(trueD/10000)
    q = np.array(testD/10000)

    # calculate m
    m = (p + q) / 2
    JSDivergence = (scipy.stats.entropy(p, m, base=2) + scipy.stats.entropy(q, m, base=2)) / 2.0
    distance = np.sqrt(JSDivergence)
    return distance

def distance(trueD, testD):
    arr = trueD - testD
    return np.linalg.norm(arr)

def constructAggreMat(node2index, fonSize):
    resultMat = np.zeros((fonSize, len(node2index)), dtype=np.float64)

    for node in node2index:
        curr = getCurrentNode(node)
        rowId = node2index[curr]
        colId = node2index[node]
        resultMat[rowId][colId] = 1.0

    return resultMat
    
def constructName(curr, prev):
    name = str(curr) + "|"
    size = len(prev)
    for i in range(size):
        pNode = prev[i]
        if(i==size-1):
            name += str(pNode)
        else:
            name += (str(pNode) + ".")
    return name

def getHoNNodeIndex(node2index, testArr, step, max_order=3):
    curr = int(testArr[step])
    prev = []
    currIndex = getNodeIndex(node2index, curr)

    for i in range(1, max_order):
        prev.append(int(testArr[step-i]))
        name = constructName(curr, prev)
        if name in node2index:
            currIndex = node2index[name]

    return currIndex

def writeDistribution(node2index, a_next, a_next_test, flowName, order, round):
    outFile = "../output_report/"+ flowName + "/distribution/" + flowName + "-" +str(order) + "-" + str(round) + "-dist.csv"

    test = []
    truth = []

    for i in range(200):
        name = str(i) + "|"

        if name in node2index:
            index = node2index[name]
            test.append(a_next_test[index])
            truth.append(a_next[index])
        else:
            test.append(0)
            truth.append(0)

    test.append(a_next_test[node2index["-1|"]])
    truth.append(a_next[node2index["-1|"]])

    with open(outFile,"w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(test)
        writer.writerow(truth)

    

def initialHoNNode(node2index, flowName, step):
    matrixSize = len(node2index)
    resultDist = np.zeros((matrixSize, 1), dtype=np.float64, order='F')
    fileName = "../input_data/" +flowName + "/"+ flowName + "-DataSequen_TESTING.csv"

    with open(fileName, 'r') as f:
        testLines = f.readlines()
        totalNum = np.float64(len(testLines))

    with open("truth.csv", 'w') as f:
        writer = csv.writer(f)
        for line in testLines:
            line = line.strip("\n")
            line = line.strip(" ")
            line = line.strip("\r")
            testArr = line.split(",")

            if testArr[0] == '':
                testArr = testArr[1:]

            if len(testArr) <= step:
                index = getNodeIndex(node2index, -1)
            else:
                index = getHoNNodeIndex(node2index, testArr, step, 3)

            index2node = convertMap(node2index)
            writer.writerow([index2node[index], index])
            resultDist[index] += 1

    return resultDist #/ totalNum

def convertMap(myMap):
    retDict = {}
    for key, item in myMap.items():
        retDict[item] = key
    return retDict

def getSourceTotalWeight(directedEdgeDict):
    totalWeight = {}
    for source, item in directedEdgeDict.items():
        totalWeight[source] = 0.0
        for _, value in item.items():
            totalWeight[source] += value
    
    return totalWeight

def constructTransMatNN(matrixSize, flowName, order):
    transitionMat = np.zeros((matrixSize, matrixSize), dtype=np.float64)
    
    fileName = "../input_data/"+flowName + "/ParaReportKLD-" + str(order) + ".csv"

    with open(fileName, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            sourceID = int(fields[0])
            targetID = int(fields[1])
            probability = float(fields[2])
            transitionMat[targetID][sourceID] = np.float64(probability)
    
    return transitionMat


