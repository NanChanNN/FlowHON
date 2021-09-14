import csv
import sys

flowName = sys.argv[1]
file = "Data/" + flowName + '/' + flowName + "-DataSequenOR.csv"

with open(file, 'r') as f:
    lines = f.readlines()
    line_num = len(lines)

resultStr = []
resultArr = []

for i in lines:
    i = i.strip("\n")
    temp = i.strip().split(" ")

    prevNode = int(temp[1])
    tempArr = [prevNode]
    tempStr = temp[0]

    for i in range(2, len(temp)):
        index = int(temp[i])
        if(index==prevNode):
            continue
        else:
            prevNode = index
            tempArr.append(prevNode)

    if(len(tempArr)<=2):
        for i in tempArr:
            tempStr += " "
            tempStr += str(i)
    else:
        for index in range(0, len(tempArr)-2):
            tempStr += " "
            tempStr += str(tempArr[index])
    
    resultStr.append(tempStr)
    resultArr.append(tempArr)
'''
outfile1 = file[:-4] + "_TRAINING.csv"
with open(outfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for s in resultStr:
        writer.writerow([s])
'''

DataDocument = "/Users/macbook/Documents/TianHe2/Nan/randomWalker/Network/"
outfile2 = DataDocument + flowName + "/" + flowName + "-DataSequenOR_Testing.csv"

with open(outfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for arr in resultArr:
        writer.writerow(arr)

print(flowName)