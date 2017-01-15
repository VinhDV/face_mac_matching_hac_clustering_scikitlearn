import os
import re
from os import listdir
import cv2
import pyrebase
from proprocess_mac import facesToMac, addData, init
import imgur_download
from preprocesser import preprocesser
from request_to_server import *
p = preprocesser()
import os
config = {
    "apiKey": "AIzaSyBNeJcW6CKp9v7z9HjRJHcOj7Sw7qY14AM",
    "authDomain": "api-server-koa.firebaseapp.com",
    "databaseURL": "https://api-server-koa.firebaseio.com",
    "storageBucket": "api-server-koa.appspot.com",
    "messagingSenderId": "409014105383"
}
counter = 0
startClassify = 300
firebase = pyrebase.initialize_app(config)


def preprocess(cur_dir):
    a = []
    c = []
    for file in listdir(cur_dir):
        if file.split(".")[-1] == "jpg":
            img = cur_dir + "/" + file
            c.append(img)
            b = p.preprocess(img)
            if b != None:
                a.append(b)
    for d in c:
        os.remove(d)
    count = 0
    for i in a:
        cv2.imwrite(cur_dir+"/"+str(count) +".jpg",i)
        count += 1


def match_mac_image(cur_dir):
    '''
    :param cur_dir: directory
            macs.txt
            1.png
            2.png
    :return:
    '''
    global startClassify
    global counter
    FaceToMac = None
    listNewFace = []
    tmp = cur_dir.split("/")
    folderName = tmp[-1]
    for f in listdir(cur_dir):
        if re.search('.jpg', f) != None:
            listNewFace.append(cur_dir + "/" + f)
    mac = open(cur_dir + "/" + 'macs.txt', 'r')
    room = None
    listNewMac = []
    alines = mac.readlines()
    lines = []
    for line in alines:
        lines.append(line.rstrip())
    for lineID in range(len(lines)):
        if lineID == 0:
            room = lines[0]
        else:
            listNewMac.append(lines[lineID])
    addData(folderName, room, listNewFace, listNewMac)
    rep = []
    if counter == startClassify:
        init()
    if counter >= startClassify:
        FaceToMac = facesToMac(listNewFace, listNewMac)
        tmp = dict()
        for facePath in FaceToMac.keys():
            if facePath == 0:
                continue
            faceID = facePath.split("/")[-1]
            ID = faceID.replace(".jpg", "")
            tmp[ID] = FaceToMac[facePath]
        for i in range(len(tmp.keys())):
            rep.append(tmp[str(i)])
    else:
        for i in range(len(listNewFace)):
            rep.append(0)
    counter += 1
    print "#frames: " + str(counter)
    return rep


def stream_handle(message):
    '''
    will be called when something new on firebase
    will be called when api/detect was called
    :param message:
    :return:
    '''
    print ('Stream handle')
    data = message["data"]
    index = message["path"]
    if (index == '/'):  # skip the first one (which is none or old data)
        return
    index = index[1:]  # remove first character
    print (index)
    print (data)

    frame = data
    cur_dir = str(index)
    print ('frame id ',cur_dir)
    os.makedirs(os.path.abspath(cur_dir))  # create new folder

    f = open(cur_dir + '/' + 'macs.txt', 'w')

    # write id device at top of macs file
    idDevice = index.split('/')[0]
    f.write(idDevice + '\n')  # write idDevice on top of macs.txt
    if (frame["macs"] != None):
        for mac in frame["macs"]:
            f.write(mac + '\n')
            print (mac)
    f.close()
    for idx, link in enumerate(frame["links"]):
        imgur_download.getImg(cur_dir, "http://125.212.233.106:3000/" + link, idx)
    preprocess(cur_dir)
    macs_result = match_mac_image(cur_dir)
    response = send_cv_result(idDevice, macs_result)  # send the result to firebase for rashberry to listen
    print (response)  # <Response [200]> is success, check firebase for result
    print ('frame id ', cur_dir)  # looking for entry with frame id on firebase for result


# macs_result = match_mac_image("/home/vdvinh/FaceNet/openface/demos/71483431485500")
# print macs_result
#preprocess("./Room1/0f823ec5-dd77-41bb-9fb8-207fdba76a3e")
#match_mac_image("./Room1/0d69c7d8-70f3-4e39-9a42-7075d68d0a47")
db = firebase.database()
my_stream = db.child("/upload").stream(stream_handler=stream_handle)

print ('breakpoint')
