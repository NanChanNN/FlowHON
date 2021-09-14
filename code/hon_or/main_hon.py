### Major update: parameter free and magnitudes faster than previous versions.
### Paper and pseudocode: https://arxiv.org/abs/1712.09658


### This package: Python implementation of the higher-order network (HON) construction algorithm.
### Paper: "Representing higher-order dependencies in networks"
### Code written by Jian Xu, Apr 2017

### Technical questions? Please contact i[at]jianxu[dot]net
### Demo of HON: please visit http://www.HigherOrderNetwork.com
### Latest code: please visit https://github.com/xyjprc/hon

### See details in README

import BuildRulesFastParameterFree
import BuildRulesFastParameterFreeFreq
import BuildNetwork
import itertools
import os
import sys
import copy
import modify
import tran
import numpy as np

LastStepsHoldOutForTesting = 0
MinimumLengthForTraining = 1
InputFileDeliminator = ' '
Verbose = False

###########################################
# Functions
###########################################

def ReadSequentialData(InputFileName):
    if Verbose:
        print('Reading raw sequential data')
    RawTrajectories = []
    with open(InputFileName) as f:
        LoopCounter = 0
        for line in f:
            fields = line.strip().split(InputFileDeliminator)
            ## In the context of global shipping, a ship sails among many ports
            ## and generate trajectories.
            ## Every line of record in the input file is in the format of:
            ## [Ship1] [Port1] [Port2] [Port3]...
            ship = fields[0]
            movements = fields[1:]

            movements.append('-1')

            LoopCounter += 1
            if LoopCounter % 10000 == 0:
                VPrint(LoopCounter)

            ## Other preprocessing or metadata processing can be added here

            ## Test for movement length
            MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
            if len(movements) < MinMovementLength:
                continue

            RawTrajectories.append([ship, movements])

    return RawTrajectories

def getTracingTime(RawTrajectories):
    nodeTracingTime = []

    for movement in RawTrajectories:
        current = []
        movement = movement[1]
        prevNode = int(movement[0])
        times = 1
        for node in movement[1:]:
            node = int(node)
            if prevNode != node:
                current.append(tuple([prevNode, times]))
                prevNode = node
                times = 0
            times+=1

        nodeTracingTime.append(current)

    return nodeTracingTime

def getRegionTime(nodeTracingTime):
    nodeMap = {}
    returnMap = {}
    for line in nodeTracingTime:
        for node in line:
            if node[0] not in nodeMap:
                nodeMap[node[0]] = []
            nodeMap[node[0]].append(node[1])

    for key, value in nodeMap.items():
        returnMap[key] = np.around(np.mean(value), 2)
    
    return returnMap

def BuildTrainingAndTesting(RawTrajectories):
    VPrint('Building training and testing')
    Training = []
    Testing = []
    for trajectory in RawTrajectories:
        ship, movement = trajectory
        movement = [key for key,grp in itertools.groupby(movement)] # remove adjacent duplications

        if LastStepsHoldOutForTesting > 0:
            Training.append([ship, movement[:-LastStepsHoldOutForTesting]])
            Testing.append([ship, movement[-LastStepsHoldOutForTesting]])
        else:
            Training.append([ship, movement])
    return Training, Testing

def DumpRules(Rules, OutputRulesFile):
    VPrint('Dumping rules to file')
    with open(OutputRulesFile, 'w', newline='') as f:
        for Source in Rules:
            for Target in Rules[Source]:
                f.write(' '.join(['.'.join([str(x) for x in Source]), '=>', Target, str(Rules[Source][Target])]) + '\n')

def DumpNetwork(Network, OutputNetworkFile):
    VPrint('Dumping network to file')
    LineCount = 0
    with open(OutputNetworkFile, 'w', newline='') as f:
        for source in Network:
            for target in Network[source]:
                f.write(','.join([SequenceToNode(source), SequenceToNode(target), str(Network[source][target])]) + '\n')
                LineCount += 1
    VPrint(str(LineCount) + ' lines written.')

def SequenceToNode(seq):
    curr = seq[-1]
    node = curr + '|'
    seq = seq[:-1]
    while len(seq) > 0:
        curr = seq[-1]
        node = node + curr + '.'
        seq = seq[:-1]
    if node[-1] == '.':
        return node[:-1]
    else:
        return node

def VPrint(string):
    if Verbose:
        print(string)


def BuildHON(InputFileName, OutputNetworkFile):
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFree.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    Network = BuildNetwork.BuildNetwork(Rules)
    DumpNetwork(Network, OutputNetworkFile)
    VPrint('Done: '+InputFileName)

def BuildHONfreq(InputFileName, OutputNetworkFile):
    print('FREQ mode!!!!!!')
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    Network = BuildNetwork.BuildNetwork(Rules)
    DumpNetwork(Network, OutputNetworkFile)
    VPrint('Done: '+InputFileName)

###########################################
# Main function
###########################################

#if __name__ == "__main__":
def top_layer(order, support, flow_name, Document, inputFile, outputFile):
    # maxOrder, miniSupport, flowName

    ## Initialize algorithm parameters
    MaxOrder = order #int(sys.argv[1])
    MinSupport = support #int(sys.argv[2])

    ## Initialize user parameters
    #InputFileName = '../../../../C2/data/synthetic/1098_ModifyMixedOrder.csv'
    FlowName = flow_name#sys.argv[3]

    #Document = "../../../HoNFlowGraph"
    Document = Document

    #InputFileName = Document + '/Data/' + FlowName + '/' + FlowName +'-DataSequen.csv'
    InputFileName = inputFile
    OutputFileName = outputFile

    #InputFileName = '../data/synthetic-major/9999.csv'
    #InputFileName = '../data/synthetic-major/1000_ModifyMixedOrder.csv'
    #InputFileName = '../data/traces-test.csv'
    #InputFileName = '../data/traces-lloyds.csv'
    OutputRulesFile = Document + '/Data/' + FlowName + '/rules-' + FlowName + '-' + str(MaxOrder) + '-' + str(MinSupport) + '_END.csv'
    OutputNetworkFile = Document + '/Data/' + FlowName + '/network-'+ FlowName + '-' + str(MaxOrder) + '-' + str(MinSupport) + '_END.csv'

    print('FREQ mode!!!!!!')
    RawTrajectories = ReadSequentialData(InputFileName)

    trajectoryCopy = copy.deepcopy(RawTrajectories)
    nodeTracingTime = getTracingTime(trajectoryCopy)

    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    DumpRules(Rules, OutputRulesFile)
    Network = BuildNetwork.BuildNetwork(Rules)

    DumpNetwork(Network, OutputNetworkFile)

    modify.writeGraphInfo(MaxOrder, MinSupport, FlowName, Document, nodeTracingTime, TrainingTrajectory)

    os.remove(OutputRulesFile)
    os.remove(OutputNetworkFile)

    tran.exchangePosition(FlowName, MaxOrder, Document, OutputFileName)

