import numpy as np

from ..lib.face_selector import ManualFaceSelector, FaceSelector
from ..util.log import Log

class FrameProcessor:

    def __init__(self, faceSelector, debug=False):
        self.log = Log("FrameProcessor")
        
        self.debug = debug
        self.faceSelector = faceSelector
        
        self.log.log("Ready")

    def getNextValues(self):
        frame = self.faceSelector.getFrame()
        if frame is not None:
            return np.mean(frame, axis=(1, 0))
        else:
            self.log.log("frame is None!")
            return None

class DummyFrameProcessor:

    def __init__(self):
        self.x = [0.0, 0.0, 0.0]
    
    def getNextValues(self):
        self.x = [i+1 for i in self.x]
        # print(f"giving:{self.x}")
        return self.x
