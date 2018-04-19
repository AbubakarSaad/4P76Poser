import argparse
import logging
import time
import ast, os, sys, csv, cv2

import common
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import matplotlib.pyplot as plt
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
outputfile = sys.path[0] + '\\a.csv'
cleanedOutputfile = sys.path[0] + '\\aClean.csv'

class dataScriptGenerator(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        self.parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
        self.parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
        self.parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
        self.args = self.parser.parse_args()
        self.scales = ast.literal_eval(self.args.scales)

        self.w, self.h = model_wh(self.args.resolution)
        self.e = TfPoseEstimator(get_graph_path(self.args.model), target_size=(self.w, self.h))

    def adHocData(self):
        directory_in_str = sys.path[0] + "\\..\\images\\OurTest\\"

        try:
            os.remove(outputfile)
            os.remove(cleanedOutputfile)
        except OSError:
            pass

        for file in os.listdir(directory_in_str):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                fullpath = directory_in_str + filename

                # estimate human poses from a single image !
                image = common.read_imgfile(fullpath, None, None)
                # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                t = time.time()
                humans = self.e.inference(image, scales=self.scales)
                elapsed = time.time() - t

                logger.info('inference image: %s in %.4f seconds.' % (fullpath, elapsed))

                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                # cv2.imshow('tf-pose-estimation result', image)
                # cv2.waitKey()

                myFile = open(outputfile, 'a')
                # myFile.write(str(filename) + ',')
                # print(filename)
                myFile.write('\n')
                myFile.close()

                try:
                    
                    fig = plt.figure()
                    a = fig.add_subplot(2, 2, 1)
                    a.set_title('Result')
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    bgimg = cv2.resize(bgimg, (self.e.heatMat.shape[1], self.e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

                    # show network output
                    a = fig.add_subplot(2, 2, 2)
                    plt.imshow(bgimg, alpha=0.5)
                    tmp = np.amax(self.e.heatMat[:, :, :-1], axis=2)
                    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
                    plt.colorbar()

                    tmp2 = self.e.pafMat.transpose((2, 0, 1))
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
                    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
                    plt.colorbar()
                    plt.show()

                    logger.info('3d lifting initialization.')
                    poseLifting = Prob3dPose(sys.path[0] + '\\lifting\\models\\prob_model_params.mat')

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
                    plt.show()

                except:
                    print ("Error when plotting image ")

        dataScriptGenerator.dataCleanup(self)


    def liveData(self):
        directory_in_str = sys.path[0] + r"/../images/LiveTest/"

        try:
            os.remove(outputfile)
            os.remove(cleanedOutputfile)
        except OSError:
            pass
       
        for file in os.listdir(directory_in_str):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                fullpath = directory_in_str + filename
                
                # estimate human poses from a single image !
                image = common.read_imgfile(fullpath, None, None)

                humans = self.e.inference(image, scales=self.scales)

                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                # cv2.imshow('tf-pose-estimation result', image)
                # cv2.waitKey()

                myFile = open(outputfile, 'a')
                # myFile.write(str(filename) + ',')
                # print(filename)
                myFile.write('\n')
                # break
                myFile.close()

    def dataCleanup(self):
        keypointData = open(outputfile, 'r')
        keypointReader = csv.reader(keypointData)
        
        cleanKeypointData = open(cleanedOutputfile, 'w', newline='')
        cleanKeypointWriter = csv.writer(cleanKeypointData)
        
        for row in keypointReader:
            badKeypointCount = 0

            #16 is the start of our x y pairs for L and R hip, knee and ankle keypoints
            for keypointIndex in range (16,28,2):

                if (row[keypointIndex] == "-1"):
                    badKeypointCount += 1
            
            if (badKeypointCount < 4):
                #good data, lets write it to our new clean file
                cleanKeypointWriter.writerow(row[:len(row)-1])

        keypointData.close()
        cleanKeypointData.close()

