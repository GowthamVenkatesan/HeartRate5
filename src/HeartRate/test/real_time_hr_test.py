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
faceSelector = FaceSelector(camera, debug=True)

log.log("Creating frameProcessor")
frameProcessor = FrameProcessor(faceSelector, debug=False)

# process 2s of video in a batch, no oeverlap
log.log("Creating batcher")
# batcher = Batcher(frameProcessor, batchSizeT=2, bufSizeT=3*2, debug=True)
# batcher = Batcher(frameProcessor, batchSizeT=4, bufSizeT=1*4, debug=False)
batcher = Batcher(frameProcessor, batchSizeT=3, bufSizeT=1*3, debug=False)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate, debug=False)

try:
    log.log("Creating runner")
    runner = Runner(batcher, hrEstimator, camera, debug=True)
except KeyboardInterrupt:
    log.log("stopping...")
finally:
    log.log("releasing camera")
    camera.release()

log.log("done")