import os, sys, csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#image directory
directory_in_str = sys.path[0] + "/tf-pose-estimation/images/Squattingpeople/"
directory = os.fsencode(directory_in_str)
print (directory)




for file in os.listdir(directory_in_str):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        
        print("Running on image: " + filename)

        command = (("python " +  sys.path[0] + "/tf-pose-estimation/src/run.py --model=cmu --resolution=432x368 --image="+sys.path[0]+"/tf-pose-estimation/images/Squattingpeople/" + filename))
        
        # print(command)
        os.system(command)

        myFile = open('example2.csv', 'a')
        myFile.write(str(filename) + ',')
        # print(filename)
        myFile.write('\n')
        # break
        
    else:
        continue