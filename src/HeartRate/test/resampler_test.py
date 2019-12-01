import os

from ..lib.camera import Camera
from ..lib.face_selector import FaceSelector, ManualFaceSelector
from ..lib.signal_processor import FrameProcessor
from ..lib.signal_util import Resampler

from ..util.file_selector import FileSelector

srcDir = r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\means"
fileSelector = FileSelector()
filePath = fileSelector.getSelectedFile(srcDir)

destFs = int(input("resampling frequency: "))
resampler = Resampler(filePath, destFs, debug=True)
resampler.resample()
