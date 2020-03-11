import cv2

from ..lib.camera import Camera
from ..lib.face_selector import DLibFaceSelector
from ..lib.image_processor import ImageProcessor

videoPath = r'D:\Gowtham\Programs\HeartRate\HeartRate5\data\gowtham.mp4'

faceSelector = DLibFaceSelector(Camera(videoPath))
frame = faceSelector.getFrame()

while frame is not None:
    ImageProcessor.renderText(frame, str(faceSelector.camera.getVideoTime()))

    cv2.imshow("From faceSelector", frame)
    frame = faceSelector.getFrame()
    cv2.waitKey(2)