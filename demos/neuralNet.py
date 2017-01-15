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

import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

np.set_printoptions(precision=2)

import openface


class encoder:
    def __init__(self, imgs='./testImage',
                 imgDim=96):
        a = os.path.realpath(__file__)
        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')
        self.dlibFacePredictor = os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.networkModel = os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7')
        self.imgDim = imgDim
        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel, self.imgDim)
        self.imgs = imgs

    def getRep(self, imgPath=None, preprpcessed=False):
        if not imgPath:
            imgPath = self.imgs
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            print("Unable to load image: {}".format(imgPath))
            return None
        if not preprpcessed:
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

            faces = self.align.getLargestFaceBoundingBox(rgbImg)
            for bb in faces:
                if bb is None:
                    print("Unable to find a face: {}".format(imgPath))
                    return None

                alignedFace = self.align.align(self.imgDim, rgbImg, bb,
                                               landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    print("Unable to align image: {}".format(imgPath))
                    return None
                bgrImg = alignedFace
                # rep.append(self.net.forward(alignedFace))
        rep = self.net.forward(bgrImg)
        return rep

    def embedFaces(self, ImgPath=None, preporcessed=False):
        if not ImgPath:
            ImgPath = self.imgs
        onlyfiles = [join(ImgPath, f) for f in listdir(ImgPath) if isfile(join(ImgPath, f))]

        faceVec = []
        name = []
        for img in onlyfiles:
            rep = self.getRep(img, preporcessed)
            if rep != None:
                faceVec.append(rep)
                name.append(img)
        return faceVec, name
