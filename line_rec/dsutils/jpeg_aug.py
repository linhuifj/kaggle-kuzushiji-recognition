import cv2
import numpy as np

class JPEGAug(object):
    def __init__(self, poss = 0.2):
        self.poss = poss * 10
        self.random_state = np.random.RandomState(None)    
    def __call__(self, img):
        if self.random_state.randint(0, 10) > self.poss:
            """ do nothing """
            return img
        
        qual = self.random_state.randint(10,100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qual]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 0)
        return img
