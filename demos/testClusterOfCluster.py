#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time

start = time.time()
from os import listdir
from os.path import isfile, join
from sklearn.cluster import AgglomerativeClustering
import argparse
import cv2
from collections import Counter,defaultdict
import os
import numpy as np
import pickle
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
#nn4.small2.v1.t7
parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    rep = []
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    path, filename = os.path.split(imgPath)
    if bgrImg is None:
        print("Unable to load image: {}".format(imgPath))
        return None
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    faces = align.getLargestFaceBoundingBox(rgbImg)
    if faces is None:
        print("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    count = 0
    for bb in faces:
        alignedFace = align.align(args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        # name = filename.split(".")[0]
        # name = name.replace("-","")
        # f_name = name + "_" + "1" +"_"+ str(count) + ".jpg"
        # cv2.imwrite(f_name, alignedFace)
        if alignedFace is None:
            print("Unable to align image: {}".format(imgPath))
        if args.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        count = count + 1
        rep.append(net.forward(alignedFace))
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep[0]


def makeTXT(ImgPath):
    onlyfiles = []
    for f in listdir(ImgPath):
        if isfile(join(ImgPath, f)) and re.search('.jpg', f) != None:
            onlyfiles.append(join(ImgPath, f))
    for pic in onlyfiles:
        path, filename = os.path.split(pic)
        name = filename.split(".")[0]
        name = name.replace("-", "")
        f_name = name + "_" + "1" + ".txt"
        g = open(f_name, "w")
        g.write("tung")
        g.close()


def processTXT(ImgPath):
    onlyfiles = []
    for f in listdir(ImgPath):
        if isfile(join(ImgPath, f)) and re.search('.txt', f) != None:
            onlyfiles.append(join(ImgPath, f))
    for text in onlyfiles:
        h = open(text, "r")
        a = h.readlines()
        path, filename = os.path.split(text)
        name = filename.split(".")[0]
        name = name.replace("_", "")
        name = name.replace("-", "")
        f_name = name + "_" + "1" + ".txt"
        g = open(f_name, "w")
        for i in a:
            g.write(i)
        g.close()
        h.close()


def processJPG(ImgPath):
    pics = []
    reps = []
    for f in listdir(ImgPath):
        if isfile(join(ImgPath, f)) and re.search('.tif', f) != None:
            pics.append(join(ImgPath, f))
    detectedPic = []
    for img in pics:
        a = getRep(img)
        if a != None:
            detectedPic.append(img)
            reps.append(getRep(img))
        a = None
    return detectedPic, reps


ImgPath = args.imgs[0]

# listClusters = defaultdict(list)
# for miniCluster in listdir(ImgPath):
#     miniCluster = ImgPath + "/" +miniCluster
#     pics, reps = processJPG(miniCluster)
#     listClusters[miniCluster].append(pics)
#     listClusters[miniCluster].append(reps)


# pickle.dump(listClusters, open( "listClusters.p", "wb" ), protocol=2)

# pics, reps = processJPG(ImgPath)
# pickle.dump(pics, open( "pics.p", "wb" ), protocol=2)
# pickle.dump(reps, open( "reps.p", "wb" ), protocol=2)

pics = pickle.load(open("pics.p", "rb"))
reps = pickle.load(open("reps.p", "rb"))
print

# def WardDistClusters(A,B):
#     pass
#
# clusters = []
# listClusters = pickle.load(open("listClusters.p", "rb"))
# clusterNameToIndex = defaultdict()
# indexToClusterName = defaultdict()
# inputClusters = defaultdict(list)
# count = 0
# for cluster in listClusters.keys():
#     inputClusters["_" + str(count)] = listClusters[cluster]
#     clusterNameToIndex[cluster] = "_" + str(count)
#     indexToClusterName["_" + str(count)] = cluster
#     count += 1
#
#
#
# for i in range(len(inputClusters)-1):
#     pass
import scipy.cluster.hierarchy as hac
from matplotlib import pyplot as plt
z = hac.linkage(reps, method="ward")
#fl = hac.fcluster(z,18,criterion='maxclust')

# plt.title('Hierarchical Clustering Dendrogram (truncated)')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# hac.dendrogram(
#     z,
#     truncate_mode='lastp',  # show only the last p merged clusters
#     p=18,  # show only the last p merged clusters
#     leaf_rotation=90.,
#     leaf_font_size=12.,
#     show_contracted=True,  # to get a distribution impression in truncated branches
# )
# plt.show()

last = z[-30:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print "clusters:", k
# from sklearn.neighbors import kneighbors_graph
#
# connectivity = kneighbors_graph(reps, n_neighbors=2, include_self=False)
# db = AgglomerativeClustering(n_clusters=18, connectivity=None,
#                              linkage='ward').fit(reps)
# labels = db.labels_
# ranking = Counter(labels).most_common(3)
#
# for hight_rank in ranking:
#     k = hight_rank[0]
#     class_member_mask = (labels == k)
#     count = 0
#     for i in range(0, len(pics)):
#         if class_member_mask[i]:
#             img = cv2.imread(pics[i])
#             count = count + 1
#             cv2.imshow("image " + str(count) + " in cluster " + str(k), img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # Number of clusters in labels, ignoring noise if present.
