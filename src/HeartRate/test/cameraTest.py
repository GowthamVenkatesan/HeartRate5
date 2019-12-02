import cv2

from ..lib.camera import Camera
from ..lib.image_processor import ImageProcessor

videoPath = r'D:\Gowtham\Programs\HeartRate\HeartRate5\data\gowtham.mp4'

camera = Camera(videoPath)
print(f"shape: {camera.getShape()}, fps: {camera.getFps()}")
frame = camera.getFrame()
while frame is not None:
    ImageProcessor.renderText(frame, str(camera.getVideoTime()))

    cv2.imshow("From camera", frame)
    frame = camera.getFrame()
    cv2.waitKey(2)

print("done")