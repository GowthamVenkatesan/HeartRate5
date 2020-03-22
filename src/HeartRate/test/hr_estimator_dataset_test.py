import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt

from ..lib.camera import Camera, DataSetCamera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.batcher import Batcher
from ..lib.hr_estimator import HREstimator, Runner
from ..lib.frame_processor import FrameProcessor
from ..util.log import Log
from ..util.file_selector import FileSelector

# videoPath = r'D:\Gowtham\Programs\HeartRate\HeartRate5\data\gowtham.mp4'

# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video1.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\mother_80hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\srini_70hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\99hr.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video2.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video3.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\charan.mp4'
# videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\charan2.mp4'

log = Log("hr_estimater_test")

DATASET_PATH = "D:\Gowtham\Programs\HeartRate\HeartRate5\data\dataset\example_data.mat"
log.log("Creating camera")
camera = DataSetCamera(DATASET_PATH, debug=False)

log.log("Creating faceSelector")
faceSelector = ManualFaceSelector(camera, debug=True)

log.log("Creating frameProcessor")
frameProcessor = FrameProcessor(faceSelector, debug=False)

# process 2s of video in a batch, no oeverlap
log.log("Creating batcher")
# batcher = Batcher(frameProcessor, batchSizeT=2, bufSizeT=3*2, debug=True)
# batcher = Batcher(frameProcessor, batchSizeT=4, bufSizeT=1*4, debug=False)
# batcher = Batcher(frameProcessor, batchSizeT=3, bufSizeT=3*1, debug=False)
batcher = Batcher(frameProcessor, batchSizeT=-1, bufSizeT=1*2, debug=True)

log.log("Creating hrEstimator")
fftWindow = camera.getFrameCount()
hrEstimator = HREstimator(batcher.getSamplingRate, fftWindow=fftWindow, debug=True)


log.log("Creating runner")
runner = Runner(batcher, hrEstimator, camera, debug=False)
log.log("stopping...")
log.log("releasing camera")
camera.release()

f = h5py.File(DATASET_PATH, "r")
ppg = f["ppg"][()][:,0]
f = plt.figure()
plt.plot(np.arange(ppg.shape[0]), ppg)
spectrum = np.fft.fft(ppg)
xf = np.fft.fftfreq(fftWindow, d=1/camera.getFps())
ppgHR = xf[ np.abs(spectrum[0:fftWindow//2-1].argmax()) ] * 60
print(f"GROUND TRUTH")
print(f"HR Estimated from PPG: {ppgHR}")

# f.close()

plt.show()
log.log("done")