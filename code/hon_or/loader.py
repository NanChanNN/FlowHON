import csv
from utilityMy import *

def loadGraphToFile(flowName, FoNNodeSet, node2index, directedEdgeDict, order, ranDocu, workLoadMapAver, workLoadMean, workLoadMapAverRegion, workLoadMapMeanRegion):
   #outFile = Docu + "/Network/"+ flowName +"/" + flowName + str(order) +"-graphOR.csv"
    #ranDocu = "../../../HoNFlowGraph/Data/"
    outFile = ranDocu + "/Data/" + flowName +"/" + flowName + str(order) +"-graphOR.csv"
    with open(outFile,"w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([len(FoNNodeSet), len(node2index)])

        for region in FoNNodeSet:
            regionName = decomposeNode(region)[0]
            writer.writerow([regionName, node2index[region]])
        
        index2node = convertMap(node2index)
        nodeWeight = getSourceTotalWeight(directedEdgeDict)

        for index in range(len(index2node)):
            HoNNode = index2node[index]

            currentNode = decomposeNode(HoNNode)[0]

            if HoNNode not in nodeWeight:
                nodeWeight[HoNNode] = 0
            HoNWeight = nodeWeight[HoNNode]
                
            writer.writerow([currentNode, HoNNode, int(HoNWeight), workLoadMapAver[HoNNode], workLoadMean[HoNNode], workLoadMapAverRegion[str(currentNode) + "|"], workLoadMapMeanRegion[str(currentNode) + "|"]])

        for source in directedEdgeDict.keys():
            for target in directedEdgeDict[source].keys():
                sourceId = node2index[source]
                targetId = node2index[target]
                value = directedEdgeDict[source][target]
                writer.writerow([sourceId, targetId, int(value)])

        theId = node2index["-1|"]
        writer.writerow([theId, theId, 1])

        print("Successfully load ", flowName)
