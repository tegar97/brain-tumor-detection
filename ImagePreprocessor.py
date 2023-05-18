import cv2 as cv
import imutils
from keras.models import load_model

class ImagePreprocessor:
    def __init__(self, target_size=(240, 240)):
        self.target_size = target_size

    def toGrayScale(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray

    def applyGaussianBlur(self, image):
        blurred = cv.GaussianBlur(image, (5, 5), 0)
        return blurred

    def applyThresholding(self,image, lower=0 , upper=255  ,default=cv.THRESH_BINARY):
        _, thresh = cv.threshold(image, lower, upper,default)
        return thresh

    def applyAdaptiveThresholding(self, image):
        thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        return thresh

    def applyErosian(self, image):
        eroded = cv.erode(image, None, iterations=2)
        return eroded

    def applyDilation(self, image):
        dilated = cv.dilate(image, None, iterations=2)
        return dilated

    def findContours(self, image):
        cnts = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def findExtremePoints(self, contours):
        c = max(contours, key=cv.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        return extLeft, extRight, extTop, extBot

    def cropAndResizeImage(self, image, extLeft, extRight, extTop, extBot):
        cropped_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        resized_image = cv.resize(cropped_image, dsize=self.target_size, interpolation=cv.INTER_CUBIC)
        normalized_image = resized_image / 255.
        reshaped_image = normalized_image.reshape((1, *self.target_size, 3))
        return reshaped_image