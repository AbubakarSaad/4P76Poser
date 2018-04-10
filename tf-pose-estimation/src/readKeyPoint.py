import csv

myData = []
 

# with myFile:
    # writer = csv.writer(myFile)
    # writer.writerows(myData)
     

def readKeyPoints(bodyIndex, bodyPartx, bodyParty):
    myFile = open('../../example2.csv', 'w')
    print("--------")
    print(bodyIndex, bodyPartx, bodyParty)
    myData.append([bodyIndex, bodyPartx, bodyParty])


def storeData():
    global myData
    i = 0
    j = 0
    while(i < 18):
        print(i, j)

        if(i < len(myData) and myData[j][0] == i):
            print(myData[i][1], myData[i][2])
            j+=1
        else:
            print("Insert new array")
        i+=1
    print(myData)



    