import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..lib.camera import Camera, RealTimeCamera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.batcher import Batcher
from ..lib.hr_estimator import HREstimator, Runner
from ..lib.frame_processor import FrameProcessor
from ..util.log import Log

log = Log("real_time_hr_test_manual_face")
videoPath = 0

log.log("Creating camera")
camera = RealTimeCamera(debug=True)

log.log("Creating faceSelector")
faceSelector = ManualFaceSelector(camera, debug=True)

log.log("Creating frameProcessor")
frameProcessor = FrameProcessor(faceSelector, debug=True)

# process 2s of video in a batch, no oeverlap
log.log("Creating batcher")
batcher = Batcher(frameProcessor, batchSizeT=1, bufSizeT=1*1, debug=True)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate, debug=False)

runner = Runner(batcher, hrEstimator, camera, debug=True)
camera.release()

log.log("done")