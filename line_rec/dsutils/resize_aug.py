import cv2
import numpy as np

class ResizeAug(object):
    def __init__(self, w, h, random_state=None, rand_scale = True):
        self.w = w
        self.h = h
        self.random_state = random_state
        self.rand_scale = rand_scale
        
        if self.random_state is None:
            self.random_state = np.random.RandomState()



    def __call__(self, img):
        imgH = img.shape[0]
        imgW = img.shape[1]
        
        if imgH == self.h and imgW == self.w:
            return img
        
        if self.rand_scale:
            scale = self.random_state.randint(-30,30)
            method = self.random_state.randint(0,2)
        else:
            scale = 0
            method = 0
            
        scale /= 100.0
        w1 = imgW * self.h / imgH
        w1 = int(w1 * (1 + scale))
        
        if w1 > self.w:
            w1 = self.w
        if method == 0:
            img = cv2.resize(img, (w1,self.h), interpolation = cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (w1,self.h), interpolation = cv2.INTER_LINEAR)
        if len(img.shape) == 2:
            img2 = np.zeros((self.h,self.w), np.uint8)
            img2[:,0:w1] = img
        else:
            img2 = np.zeros((self.h,self.w,img.shape[2]), np.uint8)
            img2[:,0:w1] = img

        return img2
