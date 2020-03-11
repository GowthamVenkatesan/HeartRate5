import os
import pickle

import numpy as np
import cv2
import dlib
from imutils import face_utils

from ..lib.camera import Camera
from ..util.models import Point, Rect
from ..util.log import Log
from ..util.file_selector import FileSelector

"""
Give us the video stream,
we'll give you only the face
"""
class FaceSelector():

    faceCascade = cv2.CascadeClassifier(r'D:\Gowtham\Programs\HeartRate\HeartRate5\src\HeartRate\lib\haarcascade_frontalface_default.xml')

    def __init__(self, camera, debug=False):
        self.log = Log("FaceSelector")

        self.debug = debug
        self.camera = camera

        # We cache the face rect and use it if we don't have any face in the cur frame
        self.faceRect = None


    def getFrame(self):
        baseImage = self.camera.getFrame()
        if baseImage is None:
            self.log.log("Got None frame!")
            return None
        grayImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        faceCood = self.faceCascade.detectMultiScale(grayImage, 1.3, 5) # scale - 1.3
        if len(faceCood) <= 0:
            self.log.log("found no faces!")
            if self.faceRect == None:
                # skip until we have found a face from start of video
                # FIXME: Skipping frames like this may cause fps calculation to be incorrect during start!!!
                self.log.log("skipping frame!")
                return self.getFrame()
        else:
            self.faceRect = Rect(Point(faceCood[0][0], faceCood[0][1]), Point(faceCood[0][0]+faceCood[0][2], faceCood[0][1]+faceCood[0][3]))
        
        selectedImage = baseImage[self.faceRect.start.y:self.faceRect.end.y, \
                                    self.faceRect.start.x:self.faceRect.end.x, \
                                    :]
    
        if self.debug:
            cv2.imshow('FaceSelector', selectedImage)
            cv2.waitKey(1000//30)
        return selectedImage

class DLibFaceSelector:
    def __init__(self, camera, maxDetectionAge=5, debug=False):
        self.log = Log("DLibFaceSelector")
        self.debug = debug
        self.camera = camera
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("D:\\Gowtham\\Programs\\HeartRate\\HeartRate5\\data\\shape_predictor_68_face_landmarks.dat")
        self.maxDetectionAge = maxDetectionAge

        self.detectionAge = 0
        self.cachedRect = None
        self.cachedContour = None
        self.cachedMask = None

        self.contourPath = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ]
    
    def getFrame(self):
        baseImage = self.camera.getFrame()
        if baseImage is None:
            return None

        # Converting the image to gray scale
        grayImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        if self.detectionAge >= self.maxDetectionAge or self.cachedRect == None:
            rects = self.detector(grayImage, 0)
            if len(rects) <= 0:
                self.log.log("found no faces!")
                if self.faceRect == None:
                    # skip until we have found a face from start of video
                    # FIXME: Skipping frames like this may cause fps calculation to be incorrect during start!!!
                    self.log.log("skipping frame!")
                    return self.getFrame()
            else:
                self.cachedRect = rects[0]
                x1, y1, x2, y2, w, h = self.cachedRect.left(), self.cachedRect.top(), self.cachedRect.right() + \
                    1, self.cachedRect.bottom() + 1, self.cachedRect.width(), self.cachedRect.height()
                shape = self.predictor(grayImage, self.cachedRect)
                self.cachedContour = face_utils.shape_to_np(shape)
                
                thisContours = self.generateContours()
                print(f"baseImage.shape: {baseImage.shape}")
                mask = np.zeros(baseImage.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, pts=[thisContours], color=255)
                # cv2.imshow("mask", mask)
                baseImage = cv2.bitwise_and(baseImage, baseImage, mask=mask)
                roi = baseImage[y1:y2, \
                                x1:x2, \
                                :]
                roi = cv2.resize(roi, (300, 300))
                mask = cv2.resize(mask, (300, 300))
                self.cachedMask = mask
                self.detectionAge = 0
        else:
            x1, y1, x2, y2, w, h = self.cachedRect.left(), self.cachedRect.top(), self.cachedRect.right() + \
                    1, self.cachedRect.bottom() + 1, self.cachedRect.width(), self.cachedRect.height()
            roi = baseImage[y1:y2, \
                                x1:x2, \
                                :]
            roi = cv2.resize(roi, (300, 300))
            roi = cv2.bitwise_and(roi, roi, mask=self.cachedMask)
            self.detectionAge += 1

        # print(f"ROI SHAPE: {roi.shape}")
        if self.debug:
            # for (x,y) in self.cachedContour:
            # x1, y1, x2, y2, w, h = self.cachedRect.left(), self.cachedRect.top(), self.cachedRect.right() + \
            #     1, self.cachedRect.bottom() + 1, self.cachedRect.width(), self.cachedRect.height()
            # cv2.circle(baseImage, (x1, y1), 5, (0, 255, 0), -1)
            # cv2.circle(baseImage, (x2, y1), 5, (255, 0, 0), -1)
            # cv2.drawContours(baseImage, [thisContours], 0, (255, 255, 255), 2)
            cv2.imshow("DLibFaceDetector", baseImage)
        return roi
    pass

    def generateContours(self):
        x1, y1, x2, y2, w, h = self.cachedRect.left(), self.cachedRect.top(), self.cachedRect.right() + \
            1, self.cachedRect.bottom() + 1, self.cachedRect.width(), self.cachedRect.height()

        thisContour = []
        # first is top left of face bounding box
        thisContour.append([x1, y1])
        # add all self.contourPath
        for i in self.contourPath:
            thisContour.append(self.cachedContour[i].tolist())
        # last is top right
        thisContour.append([x2, y1])
        return np.array(thisContour)

class ManualFaceSelector:

    def __init__(self, camera, debug=False):
        self.log = Log("ManualFaceSelector")
        
        self.camera = camera
        self.debug = debug

        self.start = None
        self.stop = None

        self.obtainBounds()


    def obtainBounds(self):
        # read first frame and obtain bounds
        frame = self.camera.getFrame()
        cmd = ord(' ')

        activeLine = 0

        self.start = (0, 0)
        self.end = (frame.shape[1], frame.shape[0])

        borderColor = (0, 0, 0)
        activeLineColor = (0, 255, 0)

        # try loading bounds if they exist
        self._loadBounds()

        while cmd != ord('e'):
            if cmd == ord('\t'):
                # toggle acitve line
                activeLine = (activeLine + 1) % 4
            elif cmd == ord('w'):
                # up
                self._moveLine(activeLine, 0, -1)
            elif cmd == ord('a'):
                # left
                self._moveLine(activeLine, -1, 0)
            elif cmd == ord('s'):
                # down
                self._moveLine(activeLine, 0, 1)
            elif cmd == ord('d'):
                # right
                self._moveLine(activeLine, 1, 0)
            elif cmd == ord('v'):
                self._saveBounds()
            elif cmd == ord('c'):
                self._loadBounds()
            elif cmd == ord('i'):
                # up
                self._moveLine(activeLine, 0, -10)
            elif cmd == ord('j'):
                # left
                self._moveLine(activeLine, -10, 0)
            elif cmd == ord('k'):
                # down
                self._moveLine(activeLine, 0, 10)
            elif cmd == ord('l'):
                # right
                self._moveLine(activeLine, 10, 0)
            else:
                self.log.log("unknown command!")

            outputImage = frame.copy()
            # draw rect
            cv2.rectangle(outputImage, self.start, self.end, borderColor, 5)
            # draw active line
            self._renderActiveLine(outputImage, activeLine, activeLineColor)
            cv2.imshow("Manual Selector:", outputImage)
            cmd = cv2.waitKey(0)
        
        # save our bounds
        self._saveBounds()
    
    def _moveLine(self, line, dx, dy):
        if line == 0:
            self.start = (self.start[0] + dx, self.start[1])
        elif line == 1:
            self.end = (self.end[0], self.end[1] + dy)
        elif line == 2:
            self.end = (self.end[0] + dx, self.end[1])
        elif line == 3:
            self.start = (self.start[0], self.start[1] + dy)
    
    def _renderActiveLine(self, img, line, lineColor):
        if line == 0:
            cv2.line(img, self.start, (self.start[0], self.end[1]), lineColor, 5)
        elif line == 1:
            cv2.line(img, (self.start[0], self.end[1]), self.end, lineColor, 5)
        elif line == 2:
            cv2.line(img, (self.end[0], self.start[1]), self.end, lineColor, 5)
        elif line == 3:
            cv2.line(img, self.start, (self.end[0], self.start[0]), lineColor, 5)

    def _logBounds(self):
        self.log.log(f"bounds: start:{self.start}, end:{self.end}")
    
    def _saveBounds(self):
        bounds = [self.start, self.end]
        saveDir = r"D:\Gowtham\Programs\HeartRate\HeartRate5\data\bounds"
        saveFileName = self.camera.videoName + ".pickle"
        saveFilePath = os.path.join(saveDir, saveFileName)
        with open(saveFilePath, 'wb') as handle:
            pickle.dump(bounds, handle)
        self.log.log('bounds saved')
    
    def _loadBounds(self):
        saveDir = r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\bounds"
        fileName = self.camera.videoName + ".pickle"
        fullPath = os.path.join(saveDir, fileName)
        try:
            with open(fullPath, 'rb') as handle:
                bounds = pickle.load(handle)
                self.start, self.end = bounds
            self.log.log('bounds loaded')
        except FileNotFoundError:
            self.log.log(f"Bounds for {self.camera.videoName} do not exist!")
        
    
    def getFrame(self):
        baseImage = self.camera.getFrame()
        if baseImage is None:
            self.log.log("Video end!")
            return None
        selectedImage = baseImage[self.start[1]:self.end[1], \
                                    self.start[0]:self.end[0], \
                                    :]
        if self.debug:
            cv2.imshow('ManualFaceSelector', selectedImage)
            cv2.waitKey(1000//30)
        return selectedImage
