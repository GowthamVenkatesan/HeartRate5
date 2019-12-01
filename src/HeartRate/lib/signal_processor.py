import time

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize

from ..lib.display import Display
from ..util.log import Log

class FrameProcessor:

    def __init__(self, faceSelector, debug=False):
        self.log = Log("FrameProcessor")
        self.debug = debug

        self.faceSelector = faceSelector
        self.log.log("Ready")

    def getNextValues(self):
        frame = self.faceSelector.getFrame()
        if frame is not None:
            return np.mean(frame, axis=(1, 0))
        else:
            self.log.log("frame is None!")
            return None

class Batcher:

    def __init__(self, frameProcessor, batchSizeT=2, bufSizeT=1*2, debug=True):
        self.frameProcessor = frameProcessor
        self.batchSize = int(batchSizeT*self.getSamplingRate())
        self.bufSize = int(bufSizeT*self.getSamplingRate())
        self.batches = self.bufSize // self.batchSize
        self.debug = debug

        # State Variables
        self.channel_means = np.empty((self.batchSize, 3))
        self.channel_means_sliding = np.empty((self.bufSize, 3))

        self.empty = True

        self.log = Log("Batcher")
    
    def getNextBatch(self):
        # self.log.log(f"onStart:{self.channel_means_sliding}")

        if self.empty:
            if self._fill_channel_means_sliding_first_run() == False:
                self.log.log("_fill_channel_means_sliding_first_run() returned False!")
                return None
            self.empty = False
            # self.log.log(f"onReturn:{self.channel_means_sliding}")
            return self.channel_means_sliding.copy()
        else:
            if not self._advance_channel_means_sliding():
                self.log.log("_advance_channel_means_sliding() returned False!")
                return None
            # self.log.log(f"onReturn:{self.channel_means_sliding}")
            return self.channel_means_sliding.copy()
    
    def _fill_channel_means(self):
        self.log.log("_fill_channel_means():")
        for i in range(self.batchSize):
            thisValues = self.frameProcessor.getNextValues()
            if thisValues is None:
                self.log.log("next values is None!")
                return False
            self.channel_means[i: ] = thisValues
        
        # for i in range(3):
        #     self.channel_means[:,i] = signal.detrend(self.channel_means[:,i])
        
        # standardize the values
        # for i in range(3):
        #     self.channel_means[:, i] /= self.channel_means[:, i].std()

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
        return np.round(self.frameProcessor.faceSelector.camera.getFps())

class IndependentComponentAnalysis:

    def __init__(self, debug=True):
        self.log = Log("IndependentComponentAnalysis")
        self.debug = debug
        self.firstRun = True

        self.ica = FastICA(n_components=3, max_iter=200)
    
    def fitTransform(self, X):
        if self.firstRun:
            self.firstRun = False
            return self.ica.fit_transform(X)
        else:
            return self.ica.transform(X)

    def getMixingMartrix(self):
        return self.ica.mixing_

class HREstimator:

    def __init__(self, fs, alpha=0.8, beta=1.2, fftWindow=512, debug=False):
        self.debug = debug
        self.log = Log("HREstimator")
        self.log.log(f"sampling rate: {fs}")

        self.alpha = alpha
        self.beta = beta
        self.lpf = LowPassFilter(fs, debug=False)
        self.bpf = BandPassFilter(fs, debug=False)
        self.ica = IndependentComponentAnalysis(debug=True)
        self.fftWindow = fftWindow
        self.T = 1/fs

        # State variables
        self.currentHR = None
    
    def estimateHR(self, x):
        """
        Given a 2d array in x
        consisting of the time series of each channel
        """
        self.thisBatchHrCache = None

        if x is None:
            self.log.log("Got None x!")
            return None

        self.log.log(f"estimateHR(); {x.mean()}")

        self.N = x.shape[0]

        # perform ICA
        self.log.log("running ica")
        x = self.ica.fitTransform(x)
        self.log.log("ica done")

        thisBatchHR = []
        for i in range(3):


            # apply lpf
            x[:,i] = self.lpf.filterSignal(x[:,i])

            # apply bpf
            # if True:
            if self.thisBatchHrCache == None:
                self.log.log("using default bpf!")
                fl = 40 / 60
                fh = 220 / 60
                x[:,i] = self.bpf.filterSignal(x[:,i], fl, fh)
            else:
                self.currentHR = self.thisBatchHrCache[i]
                fl = self.alpha*self.currentHR/60
                fh = self.beta*self.currentHR/60
                x[:,i] = self.bpf.filterSignal(x[:,i], fl, fh)
        
            yf_i = np.fft.fft(x[:,i], n=self.fftWindow)
            # xf = np.linspace(0.0, 1.0/(2.0*self.T), yf_i.shape[0]//2)
            xf = np.fft.fftfreq(self.fftWindow, d=self.T)
            hr_i = xf[ yf_i[0:self.N//2-1].argmax() ] * 60
            thisBatchHR.append(hr_i)

            # self.log.log(f"channel:{i}, hr:{hr_i}bpm")
            # self.log.log(f"thisBatchHR:{thisBatchHR}")
            # if self.debug or True:
            #     fig = plt.figure()
            #     plt.plot(xf, yf_i)
            #     plt.show()

        # for now use the green channel
        self.currentHR = thisBatchHR[1]
        self.thisBatchHrCache = thisBatchHR
        return thisBatchHR

class Runner:

    def __init__(self, batcher, hrEstimator, camera, debug=False):
        self.batcher = batcher
        self.fs = batcher.getSamplingRate()
        self.hrEstimator = hrEstimator
        self.camera = camera
        self.debug = debug

        self.log = Log("Runner")

        self.display = Display()
        self.hr = []

        thisHr = hrEstimator.estimateHR(batcher.getNextBatch())
        self.display.render(camera.getLastReadFrame(), thisHr, self.camera.getFrameNum()/self.camera.getFrameCount()*100)
        while thisHr is not None:
            self.hr.append(thisHr)
            thisHr = hrEstimator.estimateHR(batcher.getNextBatch())
            self.display.render(camera.getLastReadFrame(), thisHr, self.camera.getFrameNum()/self.camera.getFrameCount()*100)
        
        # convert hr list to np array
        self.hr = np.array(self.hr)
        
        fig = plt.figure()
        plt.suptitle("Heart Rate")
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.fs, self.hr.shape[0]), self.hr[:,0], 'blue', label='Channel 1')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.fs, self.hr.shape[0]), self.hr[:,1], 'green', label='Channel 2')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.fs, self.hr.shape[0]), self.hr[:,2], 'red', label='Channel 3')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.show()

class LowPassFilter:

    def __init__(self, fs, N = 7, fc = 5, debug=False):
    # def __init__(self, N = 2, fc = 5, fs = 30, debug=False):
        self.debug = debug
        self.filter_b, self.filter_a = signal.iirfilter(N, fc/(fs/2), btype='lowpass', analog=False)
        self.zi = None
        self.fs = fs
        self.log = Log('LowPassFilter')

        if self.debug:
            self.log.log(f'Created iir lpf, N={N}, fc={fc}, fs={fs}, b={self.filter_b}, a={self.filter_a}')
            w, h = signal.freqz(self.filter_b, self.filter_a)
            fig = plt.figure()
            plt.title('IIR LPF Frequency Response')
            ax1 = fig.add_subplot(111)
            plt.plot(w, 20 * np.log10(abs(h)), 'b')
            plt.ylabel('Amplitude [dB]', color='b')
            plt.xlabel('Frequency [rad/sample]')
            ax2 = ax1.twinx()
            angles = np.unwrap(np.angle(h))
            plt.plot(w, angles, 'g')
            plt.grid()
            plt.axis('tight')
            plt.show()
    
    def filterSignal(self, x):
        if self.zi is None:
            self.zi = signal.lfilter_zi(self.filter_b, self.filter_a)
        y, self.zi = signal.lfilter(self.filter_b, self.filter_a, x, zi=self.zi)
        if self.debug:
            figure = plt.figure()
            plt.suptitle("Low pass filter")
            plt.subplot(2,1,1)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), x)
            plt.title("raw")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.subplot(2,1,2)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), y)
            plt.title("filtered")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.show()
        return y

class BandPassFilter:

    def __init__(self, fs, N = 2, debug=False):
        self.N = N
        self.fs = fs
        self.debug = debug

        self.log = Log("BandPassFilter")
    
    def filterSignal(self, x, fl, fh):
        self.b, self.a = signal.iirfilter(self.N, [fl/(self.fs/2), fh/(self.fs/2)], btype='bandpass', analog=False)
        zi = signal.lfilter_zi(self.b, self.a)
        y, _ = signal.lfilter(self.b, self.a, x, zi=zi)

        if self.debug:
            w, h = signal.freqz(self.b, self.a)
            fig = plt.figure()
            plt.title('IIR BPF Frequency Response')
            ax1 = fig.add_subplot(111)
            plt.plot(w, 20 * np.log10(abs(h)), 'b')
            plt.ylabel('Amplitude [dB]', color='b')
            plt.xlabel('Frequency [rad/sample]')
            ax2 = ax1.twinx()
            angles = np.unwrap(np.angle(h))
            plt.plot(w, angles, 'g')
            plt.grid()
            plt.axis('tight')
            plt.show()

            figure = plt.figure()
            plt.suptitle("BandPass Filter")
            plt.subplot(2,1,1)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), x)
            plt.title("raw")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.subplot(2,1,2)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), y)
            plt.title("filtered")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.show()
        return y
