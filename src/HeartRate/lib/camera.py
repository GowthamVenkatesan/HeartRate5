from datetime import datetime
import time

import numpy as np
import cv2
import h5py

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
        return super().getFrame()

class DataSetCamera(Camera):

    def __init__(self, dataSetPath, debug=False):
        super().__init__(0, debug)
        
        self.log = Log("DataSetCamera")
        
        self.log.log("Initializing")
        self.f = h5py.File(dataSetPath, "r")
        print(self.f.keys())
        self.fs = self.f["fs"][()][0][0]
        print(f"fs: {self.fs}")
        self.ppg = self.f["ppg"][()]
        print(f"ppg: {self.ppg}")
        self.rgb = self.f["rgb"][()]
        print(f"rgb: {self.rgb}")
        print(f"rgb.shape: {self.rgb.shape}")
        
        self.frameCount = self.rgb.shape[0]
        self.currentFrame = 0

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
        if self.currentFrame < self.rgb.shape[0]:
            frame = self.f[self.rgb[self.currentFrame,0]][()]
            frame = np.swapaxes(frame, 0, 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.lastReadFrame = frame
            self.currentFrame += 1
            return frame
        else:
            self.f.close()
            return None

    def getFrameCount(self):
        return self.rgb.shape[0]
    
    def getFps(self):
        self.log.log(f"fps: {self.fs}")
        return self.fs