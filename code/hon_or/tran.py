import csv
import os 

#flowName = "computer_room"
#order = 3

def exchangePosition(flowName, order, Docu, outputFile):
    graphDocument = Docu + "/Data/"
    fileName = graphDocument + flowName + "/" + flowName + str(order) + "-graphOR.csv"

    outLineList = []
    endPosition = -1
    vertexNumber = -1
    regionNumber = -1

    with open(fileName) as f:
        fList = list(f)
        
        fields = fList[0].strip().split(',')
        regionNumber = int(fields[0])
        vertexNumber = int(fields[1])
        outLineList.append([regionNumber, vertexNumber])

        for regionCounter in range(regionNumber):
            fields = fList[1 + regionCounter].strip().split(',')
            outLineList.append(fields)

        for vertexCounter in range(vertexNumber-1):
            fields = fList[1 + regionNumber + vertexCounter].strip().split(',')

            if (int(fields[0]) == -1):
                endPosition = vertexCounter
                print(endPosition)
                fields = fList[1 + regionNumber + vertexNumber - 1].strip().split(',')           

            outLineList.append([fields[0], fields[1], fields[2]])

        outLineList.append(["-1", "-1|", 0])

        for edgeCounter in range(1 + regionNumber + vertexNumber, len(fList)):
            fields = fList[edgeCounter].strip().split(',')
            source = int(fields[0])
            target = int(fields[1])
            values = int(fields[2])

            if (source == endPosition):
                source = vertexNumber - 1
            elif (source == vertexNumber - 1):
                source = endPosition
            
            if (target == endPosition):
                target = vertexNumber - 1
            elif (target == vertexNumber - 1):
                target = endPosition

            outLineList.append([source, target, values])

    # write result
    #outFile = graphDocument + flowName +"/" + flowName + str(order) + "-graph.csv"
    outFile = outputFile

    with open(outFile,"w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        for line in outLineList:
            writer.writerow(line)

    os.remove(fileName)
