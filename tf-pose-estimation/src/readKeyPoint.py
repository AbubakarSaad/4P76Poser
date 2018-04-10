import csv

myData = []
 

# with myFile:
    # writer = csv.writer(myFile)
    # writer.writerows(myData)
     

def readKeyPoints(bodyIndex, bodyPartx, bodyParty):
    
    # print("--------")
    # print(bodyIndex, bodyPartx, bodyParty)
    myData.append([bodyIndex, bodyPartx, bodyParty])


def storeData():
    global myData
    myFile = open('/Users/abubakarsaad/Documents/MachineLearning/4P76Poser/example2.csv', 'a')
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





    