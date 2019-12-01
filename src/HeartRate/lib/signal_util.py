import os

import numpy as np
from scipy.signal import resample

from ..lib.signal_processor import FrameProcessor
from ..util.log import Log
from ..util.file_selector import FileSelector

class VideoDecoder:

    def __init__(self, frameProcessor, debug=False):
        self.frameProcessor = frameProcessor
        self.debug = debug
        self.log = Log("VideoDecoder")
    
    def decode(self):
        camera = self.frameProcessor.faceSelector.camera
        videoName = camera.videoName
        nFrames = int(camera.getFrameCount())
        self.log.log(f"FrameCount: {nFrames}")

        self.log.log(f"Creating array")
        channelMeans = np.empty((nFrames-1, 3)) # one frame is consumed by ManualFaceSelector
        self.log.log(f"Filling array")
        for i in range(nFrames):
            vals = self.frameProcessor.getNextValues()
            if vals is None:
                self.log.log("Got None!")
                break
            channelMeans[i, :] = vals[:]
        self.log.log(f"array filled")

        self.log.log(f"channelMeans: {channelMeans}")

        outputDir = r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\means"
        fileName = videoName + f".mean.dump.sampled_{int(np.round(camera.getFps()))}_hz"
        fullPath = os.path.join(outputDir, fileName)
        self.log.log(f"saving as: {fullPath}")
        np.save(fullPath, channelMeans)
        self.log.log(f"done saving")

class Resampler:

    def __init__(self, meansPath, dstFs, debug=False):
        self.meansPath = meansPath
        self.dstFs = dstFs
        
        self.log = Log("Resampler")
        self.log.log(f"file: {meansPath}, dstFs: {dstFs}")
    
    def resample(self):
        self.log.log("loading data")
        srcMeans = np.load(self.meansPath)

        # obtain fs from file name
        self.srcFs = int(self.meansPath.split(".")[-2].split("_")[1])
        self.log.log(f"detected sampling rate: {self.srcFs}hz")

        self.log.log(f"srcMeans: {srcMeans}")
        self.log.log(f"srcMeans.shape: {srcMeans.shape}")
        
        self.log.log(f"resampling")
        nResampledPts = int(srcMeans.shape[0]*self.dstFs/self.srcFs)
        dstMeans = resample(srcMeans, nResampledPts)

        self.log.log(f"resmapled signal: {dstMeans}")

        saveFileFullPath = ".".join(self.meansPath.split(".")[:-1]) + f".sampled_{self.dstFs}_hz"
        self.log.log(f"saving as: {saveFileFullPath}")
        np.save(saveFileFullPath, dstMeans)
        self.log.log(f"done")

class FileBatcher:

    def __init__(self, filePath, batchSizeT=1, bufSizeT=1*1, debug=True):
        self.filePath = filePath
        self.batchSize = int(batchSizeT*self.getSamplingRate())
        self.bufSize = int(bufSizeT*self.getSamplingRate())
        self.batches = self.bufSize // self.batchSize
        self.debug = debug
        
        self.dummyFrame = np.ones((640, 480, 3), dtype=np.uint8)

        self.log = Log("FileBatcher")
        
        self.log.log(f"Loading file: {filePath}")
        self.fullChannelMeans = np.load(filePath)
        self.log.log(f"Means loaded: {self.fullChannelMeans}")

        self.curBatch = 0
    
    def getNextBatch(self):
        self.log.log(f"getNextBatch() | curBatch: {self.curBatch}")
        if self.curBatch*self.batchSize+self.bufSize >= self.fullChannelMeans.shape[0]:
            self.log.log(f"batches over! returning None")
            return None

        thisBatch = self.fullChannelMeans[\
            (self.curBatch*self.batchSize):\
            ((self.curBatch)*self.batchSize + self.bufSize)]
        self.curBatch += 1
        return thisBatch
    
    # implemented for emulating camera intf methods, for runner
    def getLastReadFrame(self):
        return self.dummyFrame
    
    # implemented for emulating camera intf methods, for runner
    def getFrameNum(self):
        return (self.curBatch * self.batchSize) / (self.fullChannelMeans.shape[0]) * 100
    
    # implemented for emulating camera intf methods, for runner
    def getFrameCount(self):
        return self.fullChannelMeans.shape[0]
    
    def getSamplingRate(self):
        self.srcFs = int(self.filePath.split(".")[-2].split("_")[1])
        # self.log.log(f"detected sampling rate: {self.srcFs}hz")
        return self.srcFs

class NpArrayReader:

    def __init__(self):
        fileSector = FileSelector()
        filePath = fileSector.getSelectedFile(r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\means")
        arr = np.load(filePath)
        print(arr)
    