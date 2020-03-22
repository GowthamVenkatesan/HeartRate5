# got the example dataset from: https://github.com/partofthestars/PPGI-Toolbox

import h5py

DATASET_PATH = "D:\Gowtham\Programs\HeartRate\HeartRate5\data\dataset\example_data.mat"

def run():
    with h5py.File(DATASET_PATH, "r") as f:
        print(f.keys())
        fs = f["fs"].value[0][0]
        print(f"fs: {fs}")
        ppg = f["ppg"].value
        print(f"ppg: {ppg}")
    pass


run()
print("done")
