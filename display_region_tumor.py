import numpy as np
import cv2 as cv

from ImagePreprocessor import ImagePreprocessor

class DisplayTumor:
    def __init__(self, img):
        self.orig_img = np.array(img)
        self.cur_img = np.array(img)
        self.kernel = np.ones((3, 3), np.uint8)
        self.thresh = None

    def remove_noise(self):
        imagePre = ImagePreprocessor()
        gray =  imagePre.toGrayScale(self.orig_img)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.cur_img = opening


    def display_tumor(self):
        if self.thresh is None:
            self.remove_noise()

        # sure background area
        sure_bg = cv.dilate(self.cur_img, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(self.cur_img, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.orig_img, markers)
        self.orig_img[markers == -1] = [255, 0, 0]

        tumor_image = cv.cvtColor(self.orig_img, cv.COLOR_HSV2BGR)
        self.cur_img = tumor_image

    def get_current_image(self):
        return self.cur_img
