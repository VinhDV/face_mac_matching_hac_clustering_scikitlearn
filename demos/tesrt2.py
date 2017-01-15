import re
import cv2
import numpy as np
from copy import deepcopy
from neuralNet import encoder
import itertools
import math

class face:
    def __init__(self, rep, image_path):
        self.image_path = image_path
        self.rep = rep
        self.id = re.search('[0-9]{6}', self.image_path).group(0)

    def Dist(self, another_face):
        d = self.rep - another_face.rep
        return np.dot(d, d)


class Cluster:
    def __init__(self, face):
        self.list_face = []
        self.list_face.append(face)
        self.mean = None
        self.var = None

    def updateMean(self):
        sum = np.zeros(self.list_face[0].rep.shape)
        for face in self.list_face:
            sum += face.rep
        sum /= len(self.list_face)
        self.mean = sum
        return self.mean

    def updateVar(self):
        var = 0
        for face in self.list_face:
            d = face.rep - self.mean
            dif = np.dot(d, d)
            print dif
            var += dif
        self.var = var/(len(self.list_face)-1)
        std = math.sqrt(self.var)
        return std

    def meanDist(self, another_cluster):
        sum = 0
        for i in self.list_face:
            for j in another_cluster.list_face:
                sum += i.Dist(j)
        sum /= (len(self.list_face) + len(another_cluster.list_face))
        return sum

    def maxDist(self, another_cluster):
        max = 0
        imax = None
        jmax = None
        for i in self.list_face:
            for j in another_cluster.list_face:
                tmp = i.Dist(j)
                if tmp > max:
                    imax = 1
                    jmax = j
                    max = tmp
        return max, imax, jmax

    def Similar(self, another_cluster, alpha, beta):
        mean = self.meanDist(another_cluster)
        max, _, _ = self.maxDist(another_cluster)
        if mean < alpha and max < beta:
            return True, mean, max
        return False, mean, max

    def Merge(self, another_custer):
        self.list_face.extend(another_custer.list_face)

    def pprint(self, showImage = False):
        #print("mean :", self.updateMean() )
        self.updateMean()
        print("var :", self.updateVar())
        for face in self.list_face:
            if showImage:
                im = cv2.imread(face.image_path)
                cv2.imshow(face.image_path, im)
            print(face.id)
        print('\n')
        if showImage:
            cv2.waitKey(0)

class FaceSpace:
    def __init__(self, alpha, beta, imgsPath="./testImage"):
        self.encoder = encoder(imgs=imgsPath)
        self.faceRepList, self.pathlist = self.encoder.embedFaces()
        self.clusterList = []
        for i, j in zip(self.faceRepList, self.pathlist):
            self.clusterList.append(Cluster(face(i, j)))
        self.clusterListGold = deepcopy(self.clusterList)
        self._EnClusterGold()
        self.alpha = alpha
        self.beta = beta

    def _EnClusterGold(self):
        while (True):
            chaged = False
            for cluster in self.clusterListGold:
                for another_cluster in self.clusterListGold:
                    if cluster == another_cluster:
                        continue
                    if cluster.list_face[0].id == another_cluster.list_face[0].id:
                        cluster.Merge(another_cluster)
                        self.clusterListGold.remove(another_cluster)
                        chaged = True
            if not chaged:
                break

    def EnCluster(self):
        while (True):
            chaged = False
            besti, bestj = None, None
            best_mean = 5
            for i in self.clusterList:
                for j in self.clusterList:
                    if i == j:
                        continue
                    sim, mean, max = i.Similar(j, alpha=self.alpha, beta=self.beta)
                    if sim:
                        chaged = True
                        if mean < best_mean:
                            best_mean = mean
                            besti = i
                            bestj = j
            if not chaged:
                break
            besti.Merge(bestj)
            self.clusterList.remove(bestj)

    def pprint(self, showPic=False, gold=False):
        if gold:
            a = self.clusterListGold
        else:
            a = self.clusterList
        for cluster in a:
            cluster.pprint()


myFaceSpace = FaceSpace(alpha=0.3, beta=0.5)
#myFaceSpace.EnCluster()
#myFaceSpace.pprint(showPic=True, gold=True)
for cluster in myFaceSpace.clusterListGold:
    cluster.pprint()


