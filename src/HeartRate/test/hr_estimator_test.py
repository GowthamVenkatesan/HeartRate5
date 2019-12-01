import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..lib.camera import Camera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.signal_processor import Batcher, HREstimator, FrameProcessor, Runner
from ..util.log import Log
from ..util.file_selector import FileSelector

# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video1.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\mother_80hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\srini_70hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\99hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video2.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video3.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\charan.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\charan2.mp4'

log = Log("hr_estimater_test")
fileSector = FileSelector()
videoPath = fileSector.getSelectedFile(r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\video")

log.log("Creating camera")
camera = Camera(videoPath, debug=True)

log.log("Creating faceSelector")
faceSelector = ManualFaceSelector(camera, debug=True)
# faceSelector = FaceSelector(camera, debug=True)

log.log("Creating frameProcessor")
frameProcessor = FrameProcessor(faceSelector, debug=True)

# process 2s of video in a batch, no oeverlap
log.log("Creating batcher")
# batcher = Batcher(frameProcessor, batchSizeT=20, bufSize=3*20, debug=True)
batcher = Batcher(frameProcessor, batchSizeT=1, bufSizeT=2*1, debug=True)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate(), debug=False)

log.log("Creating runner")
runner = Runner(batcher, hrEstimator, camera, debug=True)
