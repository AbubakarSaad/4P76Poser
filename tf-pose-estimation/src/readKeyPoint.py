import csv
import os

myData = []
 

# with myFile:
    # writer = csv.writer(myFile)
    # writer.writerows(myData)
     

def readKeyPoints(bodyIndex, bodyPartx, bodyParty):
    
    # print("--------")
    # print(bodyIndex, bodyPartx, bodyParty)
    myData.append([bodyIndex, bodyPartx, bodyParty])


def normalizeData():
    global myData

    minX = 999999
    minY = 999999
    maxX = 0
    maxY = 0

    # get min and max vals
    for dataPiece in myData:
        if (dataPiece[1] < minX):
            minX = dataPiece[1]
        elif (dataPiece[1] > maxX):
            maxX = dataPiece[1]
        if (dataPiece[2] < minY):
            minY = dataPiece[2]
        elif (dataPiece[2] > maxY):
            maxY = dataPiece[2]
    
    # normalize all data with respect to min and max values of x and y
    for dataPiece in myData:
        # normalize X value
        dataPiece[1] = (dataPiece[1] - minX) / (maxX - minX)
        # normalize Y value
        dataPiece[2] = (dataPiece[2] - minY) / (maxY - minY)

    #done?

def storeData(numHumans, currentHuman):
    global myData
    myFile = open('a.csv', 'a')

    if (numHumans == 0):
        for i in range(36):
            myFile.write(str(-1) + ',')
    else:

        normalizeData()

        # print(myFile)
        # writer = csv.writer(myFile) 
        i = 0
        j = 0
        while(i < 18):
            # print(i, j)

            if(i < len(myData) and myData[j][0] == i):
                # print(myData[i][1], myData[i][2])
                myFile.write(str(myData[i][1]) + ',')           
                myFile.write(str(myData[i][2]) + ',')
                j+=1
            else:
                # print("Insert new array")
                myData.insert(i, [i, -1, -1])
                myFile.write(str(-1) + ',')
                myFile.write(str(-1) + ',')
                j+=1
            i+=1
            # print(myData)

        if (currentHuman < numHumans-1):
            myFile.write('\n')

    myFile.close()
    myData = []


    
