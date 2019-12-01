import cv2

from ..lib.camera import Camera
from ..lib.face_selector import FaceSelector
from ..lib.image_processor import ImageProcessor

videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video1.mp4'

faceSelector = FaceSelector(Camera(videoPath))
frame = faceSelector.getFrame()

while frame is not None:
    ImageProcessor.renderText(frame, str(faceSelector.camera.getVideoTime()))

    cv2.imshow("From faceSelector", frame)
    frame = faceSelector.getFrame()
    cv2.waitKey(2)