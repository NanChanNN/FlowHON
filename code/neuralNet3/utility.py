import numpy as np

def checkWeight(W):
    c1=0
    c2=0
    X = W[0].numpy()

    c1 = np.sum(X>0)
    c2 = np.sum(X>1)
    c3 = np.sum(X<0)
    minVal = np.min(X)
    maxVal = np.max(X)

    print("NonZeroWeight", c1, " BiggerThanOne", c2, " Smaller Than Zero", c3, "Mini value", minVal, "Max value", maxVal)

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

def getCurrentNode(nodeName):
    node = decomposeNode(nodeName)
    return (str(node[0]) + "|")

def getCurrentNodeIne(nodeName):
    node = decomposeNode(nodeName)

    return node[0]


def KLD(preHat, truth, sampleSize):
    preHat = preHat / sampleSize
    truth = truth / sampleSize

    preHat = np.clip(preHat, 1e-7, 1)
    truth = np.clip(truth, 1e-7, 1)

    return np.sum(truth * np.log(truth / preHat), axis=-1)

def manipulateMarix(matrix, manipulate, mask):
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




