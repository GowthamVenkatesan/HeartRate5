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
camera = RealTimeCamera(debug=False)

log.log("Creating faceSelector")
faceSelector = FaceSelector(camera, debug=True)

log.log("Creating frameProcessor")
frameProcessor = FrameProcessor(faceSelector, debug=False)

# process 2s of video in a batch, no oeverlap
log.log("Creating batcher")
# batcher = Batcher(frameProcessor, batchSizeT=2, bufSizeT=3*2, debug=True)
# batcher = Batcher(frameProcessor, batchSizeT=4, bufSizeT=1*4, debug=False)
batcher = Batcher(frameProcessor, batchSizeT=3, bufSizeT=3*1, debug=False)

log.log("Creating hrEstimator")
hrEstimator = HREstimator(batcher.getSamplingRate, debug=False)


log.log("Creating runner")
runner = Runner(batcher, hrEstimator, camera, debug=True)
log.log("stopping...")
log.log("releasing camera")
camera.release()

log.log("done")