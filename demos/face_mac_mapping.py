
from os import listdir
from os.path import isfile, join
from sklearn.cluster import AgglomerativeClustering
import argparse
import cv2
import os

import numpy as np

np.set_printoptions(precision=2)

from sklearn.cluster import DBSCAN

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
ImgPath = "./testImage"

align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim)

def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Unable to find a face: {}".format(imgPath))
        return None
    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        print("Unable to align image: {}".format(imgPath))
        return None
    rep = net.forward(alignedFace)
    return rep



class face:
    def __init__(self, photo_path, rep):
        self.name = photo_path
        self.rep = getRep(photo_path)


class cluster:
    def __init__(self, facelist, maclist, alpha, beta, gamma):
        self.faces = facelist
        self.macs = maclist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def maxMAC(self):
        #the MAC with maximum number of appear
        pass


    def maxClique(self):
        #largest clique with distance < gamma
        pass


    def dis1(self, B):
        # mean distant of face between A and B
        pass


    def dis2(self, B):
        # max distant of face between A and B
        pass


    def Similar(self, B):
        if self.dis1(B) < self.alpha and self.dis2(B) < self.beta:
            return True
        return False


    def Merge(self,B):
        return cluster(self.faces.extend(B.faces), self.macs.extend(B.macs))


class listCluster():
    def __init__(self, alpha, beta, gamma):
        self.clusters = []

    def add(self, newCluster):
        newCluster.maxClique()
        bestToMerge = None
        BestDist1 = 5
        for i in self.clusters:
            if newCluster.Similar(i) and newCluster.maxMAC() == i.maxMAC():
                if newCluster.dis1(i) < BestDist1:
                    bestToMerge = i
                    BestDist1 = newCluster.dis1(i)
        if bestToMerge == None:
            self.clusters.append(newCluster)
        else:
            bestToMerge.Merge(newCluster)

