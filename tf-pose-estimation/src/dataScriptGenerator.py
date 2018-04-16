import argparse
import logging
import time
import ast
import os
import sys
import csv

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
outputfile = 'a.csv'
cleanedOutputfile = 'aClean.csv'

def dataCleanup():
    print ("  CLEANING DATA...")

    keypointData = open(outputfile, 'r')
    keypointReader = csv.reader(keypointData)
    
    cleanKeypointData = open(cleanedOutputfile, 'w', newline='')
    cleanKeypointWriter = csv.writer(cleanKeypointData)
    
    for row in keypointReader:
        print (row)
        badKeypointCount = 0

        #16 is the start of our x y pairs for L and R hip, knee and ankle keypoints
        for keypointIndex in range (16,28,2):

            if (row[keypointIndex] == "-1"):
                badKeypointCount += 1
        
        if (badKeypointCount < 4):
            #good data, lets write it to our new clean file
            cleanKeypointWriter.writerow(row[:len(row)-1])

    
    print ( "DATA CLEAN COMPLETE" )
    keypointData.close()
    cleanKeypointData.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    directory_in_str = sys.path[0] + "\\..\\images\\OurTest\\"

    try:
        os.remove(outputfile)
    except OSError:
        pass

    for file in os.listdir(directory_in_str):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            fullpath = directory_in_str + filename
            
            print("Running on image: " + fullpath)

            # estimate human poses from a single image !
            image = common.read_imgfile(fullpath, None, None)
            # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            t = time.time()
            humans = e.inference(image, scales=scales)
            elapsed = time.time() - t

            logger.info('inference image: %s in %.4f seconds.' % (fullpath, elapsed))

            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            # cv2.imshow('tf-pose-estimation result', image)
            # cv2.waitKey()

            myFile = open(outputfile, 'a')
            # myFile.write(str(filename) + ',')
            # print(filename)
            myFile.write('\n')
            # break
            myFile.close()

    dataCleanup()

    sys.exit(0)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    # plt.show()

    sys.exit(0)

    logger.info('3d lifting initialization.')
    poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

    image_h, image_w = image.shape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

    for i, single_3d in enumerate(pose_3d):
        plot_pose(single_3d)
    # plt.show()

    pass
