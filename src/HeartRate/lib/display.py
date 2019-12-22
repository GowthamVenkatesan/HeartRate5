import cv2

from ..lib.image_processor import ImageProcessor
from ..util.log import Log

class Display:

    def __init__(self, fps=30, debug=False):
        self.fps = fps
        
        self.debug = debug
        self.log = Log('Display')
    
    def render(self, frame, hr, selectedHr, progress):
        output = frame.copy()
        ImageProcessor.renderBPMText(output, hr)
        ImageProcessor.renderSelectedBPMText(output, selectedHr)
        ImageProcessor.renderText(output, "%.0f%%"%(progress), pos=(10, 440))
        cv2.imshow("Display", output)
        cv2.waitKey(1000//self.fps)
