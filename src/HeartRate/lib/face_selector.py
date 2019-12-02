import os
import pickle

import numpy as np
import cv2

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
