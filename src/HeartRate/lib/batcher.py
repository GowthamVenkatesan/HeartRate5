import time

import numpy as np
from scipy import signal

from ..lib.frame_processor import FrameProcessor
from ..util.log import Log

class Batcher:

    def __init__(self, frameProcessor, batchSizeT=2, bufSizeT=1*2, debug=True):
        self.log = Log("Batcher")

        # assume you are running at 30 fps for the calculations below,
        # but you will be running at around 15 fps
        self.frameProcessor = frameProcessor
        self.batchSize = int(batchSizeT*30)
        self.bufSize = int(bufSizeT*30)
        self.batches = self.bufSize // self.batchSize
        self.debug = debug

        # State Variables
        self.channel_means = np.empty((self.batchSize, 3))
        self.channel_means_sliding = np.empty((self.bufSize, 3))

       # hamming window
        self.window = np.empty(self.channel_means_sliding.shape)
        w = np.hamming(self.channel_means_sliding.shape[0])
        for i in range(3):
            self.window[:, i] = w

        self.empty = True

        # state vars
        self.batchStartTime = None
        self.batchEndTime = None
        self.averageBatchFps = None

        # setup other modules
        # Nothing here yet...
        
        self.log.log("ready")
    
    def getNextBatch(self):
        # self.log.log(f"onStart:{self.channel_means_sliding}")

        if self.empty:
            if self._fill_channel_means_sliding_first_run() == False:
                self.log.log("_fill_channel_means_sliding_first_run() returned False!")
                return None
            self.empty = False
            # self.log.log(f"onReturn:{self.channel_means_sliding}")
            return self.channel_means_sliding.copy() * self.window
        else:
            if not self._advance_channel_means_sliding():
                self.log.log("_advance_channel_means_sliding() returned False!")
                return None
            # self.log.log(f"onReturn:{self.channel_means_sliding}")
            return self.channel_means_sliding.copy() * self.window
    
    def _fill_channel_means(self):
        self.log.log("_fill_channel_means():")
        
        # save start time
        self.batchStartTime = time.time()

        for i in range(self.batchSize):
            thisValues = self.frameProcessor.getNextValues()
            if thisValues is None:
                self.log.log("next values is None!")
                return False
            self.channel_means[i: ] = thisValues
        
        # save batch end times
        self.batchEndTime = time.time()
        # calc avgfps
        self.averageBatchFps = self.batchSize/(self.batchEndTime - self.batchStartTime)

        # detrend
        for i in range(3):
            self.channel_means[:,i] = signal.detrend(self.channel_means[:,i], overwrite_data=True)
        
        # standardize the values
        for i in range(3):
            self.channel_means[:, i] = (self.channel_means[:, i] - self.channel_means[:, i].mean()) / self.channel_means[:, i].std()

        # normalize
        # self.channel_means = normalize(self.channel_means, axis=0, norm='max')

        # self.log.log(f"mean:{self.channel_means.mean()}")

        # self.log.log(self.channel_means.shape)
        return True

    def _slide_channel_means_sliding(self):
        # print("Before Sliding:")
        # print(f"{self.channel_means_sliding}")
        self.channel_means_sliding[:-self.batchSize] = self.channel_means_sliding[self.batchSize:]
        # print("After sliding:")
        # print(f"{self.channel_means_sliding}")
    
    def _fill_channel_means_sliding_first_run(self):
        self.log.log(f"Batches:{self.batches}")
        for i in range(self.batches):
            print("[", end="")
            for j in range(i+1):
                print('#', end="")
            for j in range(self.batches-i-1):
                print(" ", end="")
            print("]")
            if self._fill_channel_means() == False:
                return
            self._slide_channel_means_sliding()
            self.channel_means_sliding[-self.batchSize:] = self.channel_means[:]
        print("Buffer full :)")
    
    def _advance_channel_means_sliding(self):
        self.log.log("_advance_channel_means_sliding():")
        _start = time.process_time()
        if self._fill_channel_means() == False:
            return False
        self._slide_channel_means_sliding()
        self.log.log(self.channel_means_sliding.shape)
        self.channel_means_sliding[-self.batchSize:, :] = self.channel_means[:, :]
        _stop = time.process_time()
        self.log.log(f"Took:{_stop-_start}s")
        # self.log.log(f"channel_means_sliding(mean):{self.channel_means_sliding.mean()}")

        # self.log.log(f"self.channel_means_sliding:{self.channel_means_sliding}")
        return True
    
    # added forom FileBatcher,
    # now Runner acquires samplingRate from us 
    def getSamplingRate(self):
        self.log.log(f"getSamplingRate(): returning: {self.averageBatchFps}")
        return self.averageBatchFps
