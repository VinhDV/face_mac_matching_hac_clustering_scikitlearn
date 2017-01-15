import numpy as np
import math
class FacePic:
    """
        A picture of face
    """
    def __init__(self, rep, frame_id, pic_dir):
        """
        :param rep: vector representation
        :param frame_id: frame this picture belong to
        :param pic_dir: directory of the picture [unique]
        """
        self.rep = rep
        self.frame_id = frame_id
        self.pic_dir = pic_dir

    def distance(self, another_facepic):
        """
        calculate distance of 2 facepics
        :param another_facepic:
        :return:
        distance of this facepic and another facepic
        """
        d = self.rep - another_facepic.rep
        d = np.dot(d,d)
        d = math.sqrt(d)
        return d
