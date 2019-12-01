import cv2

from ..lib.camera import Camera
from ..lib.image_processor import ImageProcessor

videoPath = r'C:\Users\Gowtham\Documents\programs\HeartRate\data\video1.mp4'

camera = Camera(videoPath)
print(f"shape: {camera.getShape()}, fps: {camera.getFps()}")
frame = camera.getFrame()
while frame is not None:
    ImageProcessor.renderText(frame, str(camera.getVideoTime()))

    cv2.imshow("From camera", frame)
    frame = camera.getFrame()
    cv2.waitKey(2)

print("done")