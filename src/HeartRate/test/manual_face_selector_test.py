import cv2

from ..lib.camera import Camera
from ..lib.face_selector import ManualFaceSelector
from ..util.log import Log

videoPath = r'D:\Gowtham\Programs\HeartRate\HeartRate5\data\gowtham.mp4'

log = Log("ManualFaceSelectorTest")

log.log("Creating camera")
camera = Camera(videoPath)

log.log("Creating faceSelector")
faceSelector = ManualFaceSelector(camera)
