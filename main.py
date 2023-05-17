import math
import sys
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QRadioButton, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
from keras.models import load_model
import cv2 as cv
import imutils
import numpy as np

from display_region_tumor import DisplayTumor
from predict import TumorDetector
from PyQt5 import QtCore, QtWidgets
model_path = 'best_model.h5'


class Gui(QMainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        loadUi('gui.ui', self)
        self.Image = None
        self.pushButton.clicked.connect(self.browseWindow)
        self.predict_button.clicked.connect(self.check)
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

        if(index == 0):
            self.label_2.setPixmap(pixmap)
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        else:
            self.predict_image.setPixmap(pixmap)


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
            self.label_result.setText("Tumor Detected")
            self.label_result.setStyleSheet("color: red")
        else:
            self.label_result.setText("No Tumor")
            self.label_result.setStyleSheet("color: green")

        self.displayTumor()

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


app = QApplication([])
window = Gui()
window.setWindowTitle('Module A2')
window.show()
app.exec_()
