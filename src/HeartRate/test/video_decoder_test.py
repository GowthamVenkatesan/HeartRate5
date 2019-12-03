# we don't care!

import os

from ..lib.camera import Camera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.signal_processor import FrameProcessor
from ..lib.signal_util import VideoDecoder

from ..util.file_selector import FileSelector

videoDir = r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\video"
fileSelector = FileSelector()
videoPath = fileSelector.getSelectedFile(videoDir)
print(f"opening file: {videoPath}")

camera = Camera(videoPath, debug=True)
faceSector = ManualFaceSelector(camera, debug=True)
frameProcessor = FrameProcessor(faceSector, debug=True)
videoDecoder = VideoDecoder(frameProcessor)
videoDecoder.decode()

print(f"done")
