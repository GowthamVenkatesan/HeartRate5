# As our sampling rate changes, our fft window duration in time changes,
# hold it constant in time and check if you get consistent hr estimates at different sampling frequencies


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..lib.camera import Camera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.signal_processor import Batcher, HREstimator, FrameProcessor, Runner
from ..lib.signal_util import FileBatcher
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

log = Log("hr_estimater_with_file_batcher_test")

log.log("Creating batcher")
# batcher = Batcher(frameProcessor, batchSize=20, bufSize=3*20, debug=True)
fileSelector = FileSelector()
selectedFile = fileSelector.getSelectedFile(r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\means")

# batcher = FileBatcher(selectedFile, batchSize=20, bufSize=3*20, debug=True)
batcher = FileBatcher(selectedFile, batchSizeT=1, bufSizeT=3*1, debug=True)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate(), debug=False)

log.log("Creating runner")
runner = Runner(batcher, hrEstimator, batcher, debug=True)
