import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..lib.camera import Camera, RealTimeCamera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.signal_processor import Batcher, HREstimator, FrameProcessor, Runner
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
batcher = Batcher(frameProcessor, batchSizeT=2, bufSizeT=3*2, debug=True)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate, debug=False)

try:
    log.log("Creating runner")
    runner = Runner(batcher, hrEstimator, camera, debug=True)
except KeyboardInterrupt:
    log.log("stopping...")
    camera.release()

log.log("done")