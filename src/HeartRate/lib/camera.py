from datetime import datetime
import time

import numpy as np
import cv2

from ..util.log import Log

"""
Abstraction of OpenCV Capture
video:
    0: capture from device camera
    path: path to a video file
"""
class Camera:
    
    def __init__(self, video=0, debug=False):
        self.log = Log("Camera")

        self.videoPath = video
        print(f"videoPath: {self.videoPath}")
        if video == 0:
            t = datetime.now()
            self.videoName = t.strftime("realtime__%m_%d_%Y__%H_%M_%S")
        else:
            self.videoName = video.split("\\")[-1]
        print(f"videoName: {self.videoName}")

        self.debug = debug

        self.log.log("Initializing")
        self.capture = cv2.VideoCapture(video)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera: {video}")
        self.width  = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        self.fps    = self.capture.get(cv2.CAP_PROP_FPS)           # float

        self.lastReadFrame = None

        self.log.log("Ready")
    
    def getShape(self):
        return (self.width, self.height)
    
    def getFps(self):
        self.log.log(f"fps: {self.fps}")
        return self.fps
    
    def getFrame(self):
        ret, frame = self.capture.read()
        if ret == True:
            if self.debug:
                cv2.imshow("Camera", frame)
            self.lastReadFrame = frame.copy()
            return frame
        else:
            self.log.log("Video end!")
            return None
    
    def getLastReadFrame(self):
        return self.lastReadFrame

    def getFrameNum(self):
        return self.capture.get(cv2.CAP_PROP_POS_FRAMES)
    
    def getFrameCount(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def getVideoTime(self):
        return (self.getFrameNum()/self.getFrameCount())*(self.getFrameCount()/self.getFps())

    def release(self):
        self.capture.release()


"""
Real Time Camera
exclusive for video=0
adds additional features, like dynamic fps monitoring...
"""
class RealTimeCamera(Camera):

    def __init__(self, debug=False):
        super().__init__(0, debug)
        
        self.log = Log("RealTimeCamera")
        
        self.log.log("Initializing")
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera: {0}")
        self.width  = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        self.fps    = self.capture.get(cv2.CAP_PROP_FPS)           # float

        self.lastReadFrame = None

        # state variables
        # set old time stapmp!
        self.oldFrameTimeStamp = time.time()
        self.curFrameTimeStamp = self.oldFrameTimeStamp + (1/30)
        # fps related
        self.fpsAverage = 0.0
        self.fpsAverageWindow = None

        self.log.log("Ready")
    
    def getFrame(self):
        self.oldFrameTimeStamp = self.curFrameTimeStamp
        self.curFrameTimeStamp = time.time()
        self.fpsAverage += (self.curFrameTimeStamp - self.oldFrameTimeStamp)/self.fpsAverageWindow

        # self.log.log(f"running at: {self.getFps()} fps")
        return super().getFrame()
    
    def getFps(self):
        self.log.log(f"getFps(): returning:{1/(self.curFrameTimeStamp - self.oldFrameTimeStamp)}")
        return 1/(self.curFrameTimeStamp - self.oldFrameTimeStamp)
    
    def startFpsAverageWindow(self):
        self.fpsAverage = 0.0
    
    def getFpsAverage(self):
        self.log.log(f"getFpsAverage(): returning: {self.fpsAverage}")
        return self.fpsAverage
    
    def setFpsAverageWindow(self, fpsAverageWindow):
        self.fpsAverageWindow = fpsAverageWindow