import math
import sys
from datetime import datetime

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QRadioButton, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
from keras.models import load_model
import cv2 as cv
import imutils
import numpy as np
import os
from ImagePreprocessor import ImagePreprocessor

from display_region_tumor import DisplayTumor
from predict import TumorDetector
from PyQt5 import QtCore, QtWidgets
model_path = 'best_model.h5'

from PIL import Image

class Gui(QMainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        self.thresh = None
        loadUi('gui.ui', self)
        self.Image = None
        self.ImageResult = None
        self.target_size = (224, 224)
        self.pushButton.clicked.connect(self.browseWindow)
        self.predict_button.clicked.connect(self.check)
        self.actionTransform_to_grayscale.triggered.connect(self.stepGrayscale)
        self.actionapply_gaussian.triggered.connect(self.stepGaussianBlur)
        self.actionapply_thresholding.triggered.connect(self.stepThresholding)
        self.actionapply_eroded.triggered.connect(self.stepErosion)
        self.actionapply_dilated.triggered.connect(self.stepDilation)
        self.actionfind_contours.triggered.connect(self.stepFindContours)
        self.actionTransform_to_grayscale_2.triggered.connect(self.stepGrayscale)
        self.actionApply_tresholding.triggered.connect(self.stepFindAreaTresholding)
        self.actionApply_morphology.triggered.connect(self.stepFindAreaApplyMorhology)
        self.actionApply_dilate.triggered.connect(self.stepFindAreanApplyDilate)
        self.actiondinf_foreground_area.triggered.connect(self.stepFindForeGrondArea)
        self.actionfind_uknown_region.triggered.connect(self.stepFindUnknowArea)
        self.actionmark_the_region_of_unknown_with_zero.triggered.connect(self.stepFindMarkRegion)





        self.listOfWinFrame = []

    def browseWindow(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png *.jpeg)')
        self.Image = cv2.imread(file_name)
        self.displayImage()

    def displayImage(self , index = 0 ):
        height, width, channel = self.Image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.Image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_original.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)



    def readImage(self):
        self.listOfWinFrame = []
        self.listOfWinFrame.append(self)
        self.label_result.setText("")
        self.predictTumor()

    def predictTumor(self):
        print('trigger')
        tumor_detector = TumorDetector(model_path)
        result = tumor_detector.predict_tumor(self.Image)
        if result > 0.5:
            self.label_result.setText("Brain has Tumor, Tumor Detected!")
            self.label_result.setStyleSheet("color: red")
            output_type = 'tumor'
        else:
            self.label_result.setText("Brain is healthy, No Tumor Detected")
            self.label_result.setStyleSheet("color: green")
            output_type = 'no_tumor'

        self.exportImage(output_type)
        self.displayTumor()

    def exportImage(self, output_type):
        current_date = datetime.now().strftime('%Y%m%d')

        filename = f'{output_type}_{current_date}.jpg'
        output_path = os.path.join(os.getcwd(),output_type, filename)
        cv2.imwrite(output_path, self.Image)

        print(f"Image exported to: {current_date}")

    def removeNoise(self):
        self.listOfWinFrame[0].button_view.setEnabled(True)
        self.label_result.setText("")

    def displayTumor(self):
        display_tumor = DisplayTumor(self.Image)
        display_tumor.remove_noise()
        display_tumor.display_tumor()

        tumor_image = display_tumor.get_current_image()
        height, width, channel = tumor_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(tumor_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.predict_image.setPixmap(pixmap)
        self.predict_image.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.predict_image.setScaledContents(True)


    def check(self):
        self.readImage()

    # separate step
    def stepGrayscale(self):
        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        return gray

    def stepGaussianBlur(self):
        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        #open in cv2
        cv2.imshow('blurred', blurred)
        cv2.waitKey(0)
        return blurred

    def stepThresholding(self):

        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        thresh = imagePreprocessor.applyThresholding(blurred , 45 , 255)
        #open in cv2
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
        return thresh

    def stepErosion(self):

        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        thresh = imagePreprocessor.applyThresholding(blurred , 45 , 255)
        eroded = imagePreprocessor.applyErosian(thresh)
        #open in cv2
        cv2.imshow('eroded', eroded)
        cv2.waitKey(0)
        return eroded

    def stepDilation(self):
        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        thresh = imagePreprocessor.applyThresholding(blurred , 45 , 255)
        eroded = imagePreprocessor.applyErosian(thresh)
        dilated = imagePreprocessor.applyDilation(eroded)
        # open in cv2
        cv2.imshow('dilated', dilated)
        cv2.waitKey(0)
        return dilated

    def stepFindContours(self):

        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        blurred = imagePreprocessor.applyGaussianBlur(gray)
        thresh = imagePreprocessor.applyThresholding(blurred , 45 , 255)
        eroded = imagePreprocessor.applyErosian(thresh)
        dilated = imagePreprocessor.applyDilation(eroded)
        contours = imagePreprocessor.findContours(dilated)
        extLeft, extRight, extTop, extBot = imagePreprocessor.findExtremePoints(contours)
        image = imagePreprocessor.cropAndResizeImage(self.Image, extLeft, extRight, extTop, extBot)
        # Reverse the reshape operation.
        image = image.reshape(image.shape[1], image.shape[2], image.shape[3])

        # Multiply the image by 255.
        image = image * 255

        # Convert the image to a NumPy array.
        image_array = np.array(image)

        # Convert the NumPy array to a PIL image.
        image = Image.fromarray(image_array.astype('uint8'))




        # Display the image.
        image.show()

        return image

    def stepFindAreaTresholding(self):

        imagePreprocessor = ImagePreprocessor()
        gray = imagePreprocessor.toGrayScale(self.Image)
        tresh = imagePreprocessor.applyThresholding(gray , cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


        cv2.imshow('area', tresh)
        cv2.waitKey(0)
        return tresh

    def stepFindAreaApplyMorhology(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # open in cv2
        cv2.imshow('opening', opening)

    def stepFindAreanApplyDilate(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)

        # open in cv2
        cv2.imshow('sure_bg', sure_bg)

    def stepFindForeGrondArea(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # open in cv2
        cv2.imshow('dist_transform', sure_fg)

    def stepFindUnknowArea(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # open in cv2
        cv2.imshow('unknown', unknown)

    def stepFindMarkRegion(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Image, markers)
        self.Image[markers == -1] = [255, 0, 0]

        image = cv.cvtColor(self.Image, cv.COLOR_HSV2BGR)




        # open in cv2
        cv2.imshow('image', image)












app = QApplication([])
window = Gui()
window.setWindowTitle('Tumor detector')
window.show()
app.exec_()
