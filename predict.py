import cv2 as cv
import imutils
from keras.models import load_model

from ImagePreprocessor import ImagePreprocessor


class TumorDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)




    def preprocess_image(self, image):
        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        thresh = imagePreprocessor.applyThresholding(blurred)
        erored = imagePreprocessor.applyErosian(thresh)
        dilated = imagePreprocessor.applyDilation(erored)

        cnts = imagePreprocessor.findContours(dilated)
        extLeft, extRight, extTop, extBot = imagePreprocessor.findExtremePoints(cnts)
        image = imagePreprocessor.cropAndResizeImage(image, extLeft, extRight, extTop, extBot)

        return image

    def predict_tumor(self, image):
        preprocessed_image = self.preprocess_image(image)
        res = self.model.predict(preprocessed_image)
        return res


