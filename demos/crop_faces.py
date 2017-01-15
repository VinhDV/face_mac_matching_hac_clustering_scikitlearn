__author__ = 'macpro'


import dlib
import cv2
import numpy as np
import os

current_window_name = 'Dlib'
detector = dlib.get_frontal_face_detector()

kernels = []
sharp_levels = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
for s in sharp_levels:
    kernels.append(np.array([[-s/8.0, -s/8.0, -s/8.0],
                             [-s/8.0,      s, -s/8.0],
                             [-s/8.0, -s/8.0, -s/8.0]]))


def sharpening(img):
    temp = np.zeros(img.shape)
    for i in range(0, len(kernels)):
        temp += cv2.filter2D(img, cv2.CV_32F, kernels[i])

    cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(temp)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def white_balance(img):
    img = np.log(img + 0.1) / np.log(20)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img)


def face_detection_position(src_img):
    if len(src_img.shape) > 2:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    img = cv2.equalizeHist(src_img)
    img = adjust_gamma(img, 5)

    dets = detector(img, 1)
    positions = []
    for i, d in enumerate(dets):
        positions.append(d)

    return positions


def get_faces_from_images(images_dir):
    cv2.namedWindow(current_window_name)
    for filename in os.listdir(images_dir):
        print filename
        src_img = cv2.imread(images_dir + '/' + filename)
        src_img = cv2.resize(src_img, (0, 0), fx=0.8, fy=0.8)

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        positions = face_detection_position(src_img)

        i = 0
        for d in positions:
            pt1 = (d.left(), d.top())
            pt2 = (d.right(), d.bottom())
            cv2.rectangle(src_img, pt1, pt2, (255, 0, 0), 2)
            new_img = src_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            f_name = filename + "_" + 'Face_' + str(i)
            cv2.imwrite(f_name, new_img)
            print 'Cropped - ' + f_name
            i += 1

        #cv2.imshow(current_window_name, src_img)
        #c = cv2.waitKey()
        #if c == ord('q'):
        #    break

        cv2.destroyAllWindows()


def main():
    get_faces_from_images('Photos')


if __name__ == "__main__":
    main()