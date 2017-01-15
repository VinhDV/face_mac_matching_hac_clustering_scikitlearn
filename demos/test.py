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

import time
import re
start = time.time()
from os import listdir
from os.path import isfile, join
from sklearn.cluster import AgglomerativeClustering
import argparse
import cv2
import os
import pickle

from collections import Counter, defaultdict
import numpy as np
import scipy.cluster.hierarchy as hac
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)

from sklearn.cluster import DBSCAN

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

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

def getRepPreprocessed(imgPath):
    bgrImg = cv2.imread(imgPath)
    return net.forward(bgrImg)


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
        img = cv2.imread(imgPath)
        name = filename.split(".")[0]
        cv2.imwrite("./picture/ceoTung_failed/" + name + ".jpg", img)
        return None
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    count = 0
    for bb in faces:
        alignedFace = align.align(args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        name = filename.split(".")[0]
        name = name.replace("-","")
        f_name = name + ".jpg"
        cv2.imwrite("./picture/ceoTung_processed/"+f_name, alignedFace)
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
    return rep

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


def fancy_dendrogram(*args, **kwargs):
    global subject
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hac.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title(subject)
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


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
    onlyfiles = []
    for f in listdir(ImgPath):
        if isfile(join(ImgPath, f)) and re.search('.jpg', f) != None:
            onlyfiles.append(join(ImgPath, f))
    for img in onlyfiles:
        getRep(img)

ImgPath = args.imgs[0]
#makeTXT(ImgPath)
# for i in listdir(ImgPath):
#     frame = ImgPath + "/" + i
#     for face in listdir(ImgPath + "/" + i):
#         if isfile(ImgPath + "/" + i + "/" +face) and re.search('.jpg', face) != None:
#             name = i + "_"+ face
#             img = cv2.imread(ImgPath + "/" + i + "/" +face)
#             if img != None:
#                 cv2.imwrite(name,img)


#processJPG(ImgPath)


# faceVec = []
# name = []
# for img in listdir(ImgPath):
#     rep = getRepPreprocessed(ImgPath +"/"+img)
#     faceVec.append(rep)
#     name.append(ImgPath +"/"+img)

# pickle.dump(name, open( "phuc_pic.p", "wb" ), protocol=2)
# pickle.dump(faceVec, open( "phuc_rep.p", "wb" ), protocol=2)


def demo(mac,num_clusters, num_most_common):
    if mac == "thien":
        faceVec = pickle.load(open("thien_rep.p", "rb"))
        name = pickle.load(open("thien_pic.p", "rb"))
    elif mac == "tung":
        faceVec = pickle.load(open("tung_rep.p", "rb"))
        name = pickle.load(open("tung_pic.p", "rb"))
    elif mac == "phuc":
        faceVec = pickle.load(open("phuc_rep.p", "rb"))
        name = pickle.load(open("phuc_pic.p", "rb"))
    elif mac == "chi":
        faceVec = pickle.load(open("chi_rep.p", "rb"))
        name = pickle.load(open("chi_pic.p", "rb"))
    elif mac == "ceoTung":
        faceVec = pickle.load(open("ceoTung_rep.p", "rb"))
        name = pickle.load(open("ceoTung_pic.p", "rb"))
    else:
        print ("mac not exist")
        return

    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(faceVec, n_neighbors=2, include_self=False)
    db = AgglomerativeClustering(n_clusters=num_clusters, connectivity=None,
                                 linkage='ward').fit(faceVec)

    labels = db.labels_
    ranking = Counter(labels).most_common(num_most_common)

    for hight_rank in ranking:
        k = hight_rank[0]
        class_member_mask = (labels == k)
        count = 0
        for i in range(0, len(name)):
            if class_member_mask[i]:
                img = cv2.imread(name[i])
                count = count + 1
                cv2.imshow("image " + str(count) + " in cluster " + str(k), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






#demo(mac="ceoTung",num_clusters=4,num_most_common=4)
macs = ['ceoTung','chi','thien','tung','phuc']
mac_to_clusters = dict()
mac_to_path = dict()
for mac in macs:
    rep_file_name = mac + "_rep.p"
    pic_file_name = mac + "_pic.p"
    mac_to_clusters[mac] = pickle.load(open(rep_file_name, "rb"))
    mac_to_path[mac] = pickle.load(open(pic_file_name, "rb"))

subject = "tung"
z = hac.linkage(mac_to_clusters[subject], method="ward")
name = mac_to_path[subject]
plt.figure(figsize=(10,10))
fancy_dendrogram(
    z,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=40,
    max_d=1.75,
)
plt.show()

# for subject in macs:
#     z = hac.linkage(mac_to_clusters[subject], method="ward")
#     name = mac_to_path[subject]
#
#     plt.figure(figsize=(10,10))
#     fancy_dendrogram(
#         z,
#         leaf_rotation=90.,
#         leaf_font_size=12.,
#         show_contracted=True,
#         annotate_above=40,
#         # max_d=1.3,
#     )
#     plt.show()
show_face = True
if show_face:
    max_d = 1.75
    labels = hac.fcluster(z, max_d, criterion='distance')
    ranking = Counter(labels).most_common(2)

    for hight_rank in ranking:
        k = hight_rank[0]
        class_member_mask = (labels == k)
        count = 0
        for i in range(0, len(name)):
            if class_member_mask[i]:
                img = cv2.imread(name[i])
                count = count + 1
                cv2.imshow("image " + str(count) + " in cluster " + str(k), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

