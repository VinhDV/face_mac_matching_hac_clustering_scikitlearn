import os
import pickle
from collections import defaultdict, Counter
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import scipy
import scipy.cluster.hierarchy as hac
from munkres import Munkres
from scipy.spatial.distance import mahalanobis

from neuralNet import encoder

fileDir = os.path.dirname(os.path.realpath(__file__))
mac1 = fileDir + "/testImage/Room1_preprocess/mac"
mac2 = fileDir + "/testImage/Room2_preprocess/mac"

face1 = fileDir + "/testImage/Room1_preprocess/faces"
face2 = fileDir + "/testImage/Room2_preprocess/faces"

thien = fileDir + "/testImage/Thien"
vinh = fileDir + "/testImage/Vinh"
tung = fileDir + "/testImage/Tung"
minh = fileDir + "/testImage/Minh"

MacToPicGold = defaultdict(list)
for f in listdir(thien):
    MacToPicGold['thien'].append(thien + "/" + f)
for f in listdir(vinh):
    MacToPicGold['vinh'].append(vinh + "/" + f)
for f in listdir(tung):
    MacToPicGold['tunh'].append(tung + "/" + f)
for f in listdir(minh):
    MacToPicGold['minh'].append(minh + "/" + f)

minimumPics = 10
close = 0.6
MERGE_THRESHOLD = 1.75
RECOGNISE_THRESHOLD = 23
E = encoder()
macToPic = None
picToRep = None
host = None


def isPSD(A, tol=1e-8):
    E, V = scipy.linalg.eigh(A)
    return np.all(E > -tol)


def FindStationary(MacOfFilename):
    stations = MacOfFilename[MacOfFilename.keys()[0]]
    for i in MacOfFilename.keys():
        stations = stations.intersection(MacOfFilename[i])
    return stations


def getMac(ImgPath):
    room = []
    onlyfiles = []
    MacOfFilename = defaultdict(set)
    for f in listdir(ImgPath):
        if isfile(join(ImgPath, f)):
            onlyfiles.append(join(ImgPath, f))
    for text in onlyfiles:
        h = open(text, "r")
        a = h.readlines()
        rep = []
        for i in a:
            rep.append(i.replace("\r\n", "").replace("\n", ""))
        room.append(set(rep))
        path, filename = os.path.split(text)
        photo_name = filename.split("_")[0]
        MacOfFilename[photo_name] = MacOfFilename[photo_name].union(set(rep))
        h.close()
    stations = FindStationary(MacOfFilename)
    for station in stations:
        for i in MacOfFilename.keys():
            MacOfFilename[i].remove(station)
    for photo in room:
        for station in stations:
            photo.remove(station)
    return room, MacOfFilename


def pos_customers(mac1, mac2):
    room1, MacOfFile_room1 = getMac(mac1)
    allMac1 = set()
    room2, MacOfFile_room2 = getMac(mac2)
    allMac2 = set()

    for i in room1:
        allMac1 = allMac1.union(i)
    for i in room2:
        allMac2 = allMac2.union(i)
    customer_macs = allMac1.intersection(allMac2)
    return customer_macs, MacOfFile_room1, MacOfFile_room2


def invert(d):
    ret = defaultdict(set)
    for key, values in d.items():
        for value in values:
            ret[value].add(key)
    return ret


def customerMacWithFace(mac1, mac2):
    MacToFaceNames = defaultdict(set)
    customer_macs, macOfFileRoom1, macOfFileRoom2 = pos_customers(mac1, mac2)
    facesAppearWithMac1 = invert(macOfFileRoom1)
    facesAppearWithMac2 = invert(macOfFileRoom2)
    for customer in customer_macs:
        MacToFaceNames[customer] = MacToFaceNames[customer].union(facesAppearWithMac1[customer])
        MacToFaceNames[customer] = MacToFaceNames[customer].union(facesAppearWithMac2[customer])

    for customer in MacToFaceNames.keys():
        if len(MacToFaceNames[customer]) < minimumPics:
            MacToFaceNames.pop(customer)
    return MacToFaceNames


def macToFacePath(mac1, mac2, face1, face2):
    rep = defaultdict(set)
    customer_PhotoName = customerMacWithFace(mac1, mac2)
    FacePath1 = [join(face1, f) for f in listdir(face1) if isfile(join(face1, f))]
    FacePath2 = [join(face2, f) for f in listdir(face2) if isfile(join(face2, f))]
    for customerMac in customer_PhotoName.keys():
        for name in customer_PhotoName[customerMac]:
            for path in FacePath1:
                os_path, faceName = os.path.split(path)
                if name == faceName.split("_")[0]:
                    rep[customerMac].add(path)
            for path in FacePath2:
                os_path, faceName = os.path.split(path)
                if name == faceName.split("_")[0]:
                    rep[customerMac].add(path)
    return rep


def pp(matrix):
    for i in matrix.keys():
        l = []
        for j in matrix[i]:
            l.append(matrix[i][j])
        print l


def findCenter(matrix):
    curCenter = None
    curNumberOfSupport = -1
    for i in matrix.keys():
        l = []
        for j in matrix[i]:
            l.append(matrix[i][j])
        tmp = sum(l)
        if tmp > curNumberOfSupport:
            curNumberOfSupport = tmp
            curCenter = i
    return curCenter, curNumberOfSupport


def showCenterAndSupport(center, voter):
    """
    use: showCenterAndSupport(center, voter[center])
    """
    img = cv2.imread(center)
    cv2.imshow("center", img)
    for support in voter:
        if voter[support] == 1:
            img = cv2.imread(support)
            cv2.imshow(support, img)
            cv2.waitKey(0)
    cv2.waitKey(0)


def bestClusterInMAC(pic_list, picToRep):
    pics = list(pic_list)
    reps = []
    host_cluster = []
    for pic in pics:
        reps.append(picToRep[pic])
    z = hac.linkage(reps, method="ward")
    labels = hac.fcluster(z, MERGE_THRESHOLD, criterion='distance')
    best = Counter(labels).most_common(1)
    k = best[0][0]
    class_member_mask = (labels == k)
    for i in range(0, len(pics)):
        if class_member_mask[i]:
            host_cluster.append(pics[i])
    return host_cluster


def showAllFaceInMac(mac, pics):
    for pic in pics:
        img = cv2.imread(pic)
        cv2.imshow(pic, img)
        cv2.waitKey(200)
    cv2.waitKey(0)


# the number of photo of a person recognised < number of time their MAC appear
def MacToFace(mac1, mac2, face1, face2):
    global macToPic
    global picToRep
    macToPic = macToFacePath(mac1, mac2, face1, face2)
    picToRep = defaultdict(set)
    for mac in macToPic.keys():
        for pic in macToPic[mac]:
            picToRep[pic] = E.getRep(pic, preprpcessed=True)
    host = defaultdict(list)
    for mac in macToPic.keys():
        host[mac] = bestClusterInMAC(macToPic[mac], picToRep)
    return host, picToRep, macToPic


def init():
    global host
    global picToRep
    global macToPic
    print "Initializing..."
    # host, picToRep, macToPic = MacToFace(mac1, mac2, face1, face2)
    # pickle.dump(host,open( "host.p", "wb" ), protocol=2)
    # pickle.dump(picToRep, open("picToRep.p", "wb"), protocol=2)
    # pickle.dump(macToPic, open("macToPic.p", "wb"), protocol=2)
    host = pickle.load(open("host.p", "rb"))
    picToRep = pickle.load(open("picToRep.p", "rb"))
    macToPic = pickle.load(open("macToPic.p", "rb"))
    print "Done initialization"


# searchingMac = 'tung'
# center, voter, dist = bestPicInMAC(macToPic[searchingMac], picToRep)
# print "-----------------------showCenterAndSupport--------------------------------"
# showCenterAndSupport(center, voter[center])

# cv2.destroyAllWindows()
# print "-------------------------showAllFaceInMac----------------------------------"
# showAllFaceInMac(searchingMac, macToPic[searchingMac])
# cv2.destroyAllWindows()

# print "-----------------------Show Cluster in MAC--------------------------------"
# from sklearn.neighbors import kneighbors_graph
# faceVec = []
# path= []
# for pic in macToPic[searchingMac]:
#     faceVec.append(picToRep[pic])
#     path.append(pic)
#
# connectivity = kneighbors_graph(faceVec, n_neighbors=2, include_self=False)
# db = AgglomerativeClustering(n_clusters=5, connectivity=connectivity,
# db = AgglomerativeClustering(n_clusters=5, connectivity=connectivity,
#                              linkage='ward').fit(faceVec)
# labels = db.labels_
#
# for k in np.unique(labels):
#     class_member_mask = (labels == k)
#     count = 0
#     for i in range(0, len(path)):
#         if class_member_mask[i]:
#             img = cv2.imread(path[i])
#             count = count + 100
#             cv2.imshow("image" + str(count) + "in cluster" + str(k), img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#



# macToPic = macToFacePath(mac1, mac2, face1, face2)
# showAllFaceInMac('vinh', macToPic['vinh'])

def distVect(vec1, vec2):
    d = vec1 - vec2
    return np.dot(d, d)


def pad_to_square(a, pad_value=0):
    m = a.reshape((a.shape[0], -1))
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
    padded[0:m.shape[0], 0:m.shape[1]] = m
    return padded


def varCluster(cluster, picToRep, meanCluster):
    var = 0
    for pic in cluster:
        var += distVect(picToRep[pic], meanCluster)
    return var / (len(cluster) - 1)


def meanCluster(cluster, picToRep):
    sum = 0
    for pic in cluster:
        sum += picToRep[pic]
    return sum / len(cluster)


def nomalizeDist(cluster, inputPic, picToRep):
    sumdist = 0
    reps = []
    for pic in cluster:
        reps.append(picToRep[pic])
    clus = np.vstack(reps)
    clus = np.transpose(clus)
    cov = np.cov(clus)
    if np.linalg.cond(cov) > np.finfo(cov.dtype).eps:
        inv_cov = np.linalg.pinv(cov)
    else:
        inv_cov = np.linalg.inv(cov)

    y = picToRep[inputPic]
    mean_cluster = meanCluster(cluster, picToRep)
    d = mahalanobis(y, mean_cluster, inv_cov)
    return d


def CusterOfMAC(centers):
    """
    :param centers
    :return: dict[mac] = [list of voters]
    """
    ret = defaultdict()
    for mac in centers:
        center = centers[mac][0]
        voter = []
        voter.append(center)
        for pic in centers[mac][1][center]:
            if centers[mac][1][center][pic] == 1:
                voter.append(pic)
        ret[mac] = voter
    return ret


def addData(FolderName, RoomID, listNewFace, listNewMac):
    facePath = None
    macPath = None

    if RoomID == 'Room1':
        macPath = mac1
        facePath = face1
    elif RoomID == 'Room2':
        macPath = mac2
        facePath = face2

    for face in listNewFace:
        img = cv2.imread(face)
        path, filename = os.path.split(face)
        filename = FolderName + "_" + filename
        fullFileName = facePath + "/" + filename
        cv2.imwrite(fullFileName, img)

    macName = FolderName + ".txt"
    fullMacName = macPath + "/" + macName
    f = open(fullMacName, "w")
    for newMac in listNewMac:
        f.write(newMac + "\n")


def facesToMac(listNewFace, listNewMac):
    global host
    global picToRep
    global macToPic

    for newMac in listNewMac:
        for newface in listNewFace:
            macToPic[newMac].add(newface)

    for newface in listNewFace:
        picToRep[newface] = E.getRep(newface, preprpcessed=True)

    clusterOfMac = defaultdict()
    for i in host.keys():
        if i in listNewMac:
            clusterOfMac[i] = host[i]

    clusterOfMacToIndex = defaultdict()
    IndexToClusterOfMac = defaultdict(lambda: -1)
    listNewFaceToIndex = defaultdict()
    IndexTolistNewFace = defaultdict(lambda: 0)

    count = 0
    for mac in clusterOfMac:
        clusterOfMacToIndex[mac] = count
        IndexToClusterOfMac[count] = mac
        count += 1

    count = 0
    for face in listNewFace:
        listNewFaceToIndex[face] = count
        IndexTolistNewFace[count] = face
        count += 1

    NewFacesToClustersDist = np.zeros((len(clusterOfMac.keys()), len(listNewFace)))
    for mac in clusterOfMac:
        for face in listNewFace:
            NewFacesToClustersDist[clusterOfMacToIndex[mac]][listNewFaceToIndex[face]] \
                = nomalizeDist(clusterOfMac[mac], face, picToRep)
    # idx = 0
    for i in range(NewFacesToClustersDist.shape[0]):
        if np.all(NewFacesToClustersDist[i] > RECOGNISE_THRESHOLD):
            # NewFacesToClustersDist = np.delete(NewFacesToClustersDist,idx,0)
            # IndexToClusterOfMac.pop(IndexToClusterOfMac.keys()[idx])
            # idx = idx-1
            for j in range(NewFacesToClustersDist.shape[1]):
                NewFacesToClustersDist[i][j] = 987654321
                # idx += 1

    m = Munkres()
    NewFacesToClustersDist = pad_to_square(NewFacesToClustersDist, pad_value=0)
    tmp = NewFacesToClustersDist.tolist()
    indexes = m.compute(tmp)

    ret = defaultdict()
    print "MATCH:"
    for row, column in indexes:
        if tmp[row][column] == 987654321:
            IndexToClusterOfMac[row] = -1
        ret[IndexTolistNewFace[column]] = IndexToClusterOfMac[row]
        print "\t" + str(IndexTolistNewFace[column]) + " : " + str(IndexToClusterOfMac[row]) + " (dist: " + str(
            tmp[row][column]) + ")"
    return ret

# host = pickle.load(open("host.p", "rb"))
# picToRep = pickle.load(open("picToRep.p", "rb"))
# macToPic = pickle.load(open("macToPic.p", "rb"))
# for pic in host['minh']:
#     img = cv2.imread(pic)
#     cv2.imshow(pic,img)
# cv2.waitKey(0)