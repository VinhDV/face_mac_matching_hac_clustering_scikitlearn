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
import openface


class preprocesser:
    def __init__(self, imgDim=96):
        a = os.path.realpath(__file__)
        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')
        self.dlibFacePredictor = os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.imgDim = imgDim
        self.align = openface.AlignDlib(self.dlibFacePredictor)

    def preprocess(self,img_path):
        bgrImg = cv2.imread(img_path)
        path, filename = os.path.split(img_path)
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        faces = self.align.getLargestFaceBoundingBox(rgbImg)
        if faces != None:
            face = faces[0]
        else:
            face = None
        if face is None:
            print("Unable to find a face: {}".format(img_path))
            return None
        alignedFace = self.align.align(self.imgDim, rgbImg, face,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Unable to align image: {}".format(img_path))
            return None
        return alignedFace