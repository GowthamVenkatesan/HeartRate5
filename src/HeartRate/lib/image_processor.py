import numpy as np
import time
import cv2

from matplotlib import pyplot as plt

"""
Whole-frame image processing components & helper methods
"""
class ImageProcessor:

    # Path to the facecascade
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    @classmethod
    def getGrayscale(cls, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    @classmethod
    def getHsv(cls, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    @classmethod
    def equalizeImage(cls, img):
        pass

    @classmethod
    def _drawHistogram(cls, img, channels, color='k'):
        hist = cv2.calcHist([img], channels, None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    
    @classmethod
    def drawColorHistogram(cls, img):
        for i, col in enumerate(['b', 'g', 'r']):
            cls._drawHistogram(img, [i], color=col)
        plt.show()
    
    @classmethod
    def drawGrayscaleHistogram(cls, img):
        grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cls._drawHistogram(grayscaleImage, [0])
        plt.show()

    @classmethod
    def renderText(cls, img, text, pos=(10, 10), textSize=2):
        col = (255, 0, 0)
        cv2.putText(img, text, \
                    pos, cv2.FONT_HERSHEY_PLAIN, textSize, col)

    @classmethod
    def renderBPMText(cls, img, bpm):
        if bpm is None:
            return

        tsize = 2
        text = "%.0fbpm %.2fhz" % (bpm[0], bpm[0]/60)
        cv2.putText(img, text, \
            (10, 30), cv2.FONT_HERSHEY_PLAIN, tsize, (30, 30, 255))
        text = "%.0fbpm %.2fhz" % (bpm[1], bpm[1]/60)
        cv2.putText(img, text, \
            (10, 70), cv2.FONT_HERSHEY_PLAIN, tsize, (30, 255, 30))
        text = "%.0fbpm %.2fhz" % (bpm[2], bpm[2]/60)
        cv2.putText(img, text, \
            (10, 110), cv2.FONT_HERSHEY_PLAIN, tsize, (255, 30, 30))
    
    @classmethod
    def renderSelectedBPMText(cls, img, bpm):
        if bpm is None:
            return

        tsize = 2
        text = "%.0fbpm %.2fhz" % (bpm, bpm/60)
        cv2.putText(img, text, \
                    (10, 150), cv2.FONT_HERSHEY_PLAIN, tsize, (255, 255, 255))