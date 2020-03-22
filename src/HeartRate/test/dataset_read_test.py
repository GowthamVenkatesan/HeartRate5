# got the example dataset from: https://github.com/partofthestars/PPGI-Toolbox

import h5py
import numpy as np
import cv2

DATASET_PATH = "D:\Gowtham\Programs\HeartRate\HeartRate5\data\dataset\example_data.mat"

def run():
    with h5py.File(DATASET_PATH, "r") as f:
        print(f.keys())
        fs = f["fs"][()][0][0]
        print(f"fs: {fs}")
        ppg = f["ppg"][()]
        print(f"ppg: {ppg}")
        rgb = f["rgb"][()]
        print(f"rgb: {rgb}")
        print(f"rgb.shape: {rgb.shape}")
        for i in range(rgb.shape[0]):
            frame = f[rgb[i,0]][()]
            frame = np.swapaxes(frame, 0, 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(f"frame: {frame}")
            print(f"frame.shape: {frame.shape}")
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
    pass


run()
print("done")
