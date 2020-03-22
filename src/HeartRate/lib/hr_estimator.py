import datetime

import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from ..lib.signal_processor import IndependentComponentAnalysis, LowPassFilter, BandPassFilter
from ..lib.display import Display
from ..util.log import Log

class ChannelSelctor:

    def __init__(self):
        self.log = Log("ChannelSelector")
        self.log.log("Ready")

        # TODO: Set the limits correctly
        # Values used now are to be changed in fututre!!!
        self.l_limit = 60
        self.h_limit = 120
    
    def select(self, values):
        self.log.log(f"selct():")
        selectedValues = [val for val in values if val < self.h_limit and val > self.l_limit]
        if len(selectedValues) == 0:
            # FATAL, None of the selected values contain the heart rate
            # work with the second op from ica
            self.log.log(f"FATAL: None of the channels contain hr, using second val, returning {values[1]}")
            return values[1]
        res = sum(selectedValues)/len(selectedValues)
        self.log.log(f"values: {values} selected: {selectedValues} returning: {res}")
        return res

class MaxSpectrumChannelSelector:

    def __init__(self):
        self.log = Log("MaxSpectrumChannelSelector")
        self.log.log("Ready")

    def select(self, spectra):
        self.log.log(f"select():")
        # maxPowers = [spectra[:,i].max() for i in range(3)]
        maxPowers = np.amax(np.abs(spectra), axis=0)
        self.log.log(f"maxPowers: {maxPowers}")
        return spectra[:,np.argmax(maxPowers)]

class HREstimator:

    def __init__(self, getFs, alpha=0.8, beta=1.2, fftWindow=512, debug=False):
        self.debug = debug
        self.log = Log("HREstimator")
        self.log.log(f"sampling rate: {getFs()}")

        # alpha & beta values for bpf
        self.alpha = alpha
        self.beta = beta
        
        # Filter creation:
        # the filters will be created for each batch,
        # to accomodate for the varying sampling rates...

        # ica
        self.ica = IndependentComponentAnalysis(debug=self.debug)

        # channelSelector
        self.channelSelector = MaxSpectrumChannelSelector()

        # length of fft
        self.fftWindow = fftWindow

        # cache of method to get fs
        self.getFs = getFs

        # State variables
        self.currentHR = None

        self.colors = [ "r", "g", "b" ]
    
    def estimateHR(self, x):
        """
        Given a 2d array in x
        consisting of the time series of each channel
        """
        if x is None:
            self.log.log("Got None x!")
            return None

        self.N = x.shape[0]
        # alloc array for holding spectrum
        spectra = np.empty((self.fftWindow, 3), dtype=np.complex)

        self.log.log("running ica")
        x = self.ica.fitTransform(x)
        self.log.log("ica done")

        thisBatchHR = []

        self.lpf = LowPassFilter(self.getFs(), debug=self.debug, N=7)
        self.bpf = BandPassFilter(self.getFs(), debug=self.debug, N=2)

        if self.debug:
            fig = plt.figure("HR Estimator")
            plt.clf()
        
        for i in range(3):
            x[:,i] = self.lpf.filterSignal(x[:,i])
            if self.currentHR == None:
                self.log.log("using default bpf!")
                fl = 40 / 60
                fh = 220 / 60
                x[:,i] = self.bpf.filterSignal(x[:,i], fl, fh)
            else:
                # FIXME The selected HR of the 3 values should be used,
                # this approach causes loosing a signal completely if it goes to 0 once, due to the preserved filtering!
                fl = self.alpha*self.currentHR/60
                fh = self.beta*self.currentHR/60
                x[:,i] = self.bpf.filterSignal(x[:,i], fl, fh)
            
            yf_i = np.fft.fft(x[:,i], n=self.fftWindow)
            spectra[:, i] = yf_i
            self.log.log(f"y_{i}: \n{yf_i[:2]}")

            xf = np.fft.fftfreq(self.fftWindow, d=1/self.getFs())
            hr_i = xf[ yf_i[0:self.fftWindow//2-1].argmax() ] * 60
            thisBatchHR.append(hr_i)

            if self.debug:
                fig = plt.figure("HR Estimator")
                plt.plot(xf[0:self.fftWindow//2-1], np.abs(yf_i[0:self.fftWindow//2-1])**2, self.colors[i])
                plt.draw()
                plt.pause(0.001)

        # perform channel selection
        self.log.log(f"spectra: \n{spectra[:2, :]}")

        # select the channel using the ChannelSelctor
        selectedSpectrum = self.channelSelector.select(spectra)
        xf = np.fft.fftfreq(self.fftWindow, d=1/self.getFs())
        self.currentHR = xf[ selectedSpectrum[0:self.fftWindow//2-1].argmax() ] * 60
        # thisBatchHR.append(hr_i)
        # self.currentHR = self.channelSelector.select(thisBatchHR)
        return thisBatchHR, self.currentHR

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
        self.selectedHr = []
        
        self.executor = ThreadPoolExecutor(max_workers=1)

        # run
        self.loop()
    
    def loop(self):
        thisHr = 0
        try:
            while thisHr is not None:
                t = datetime.datetime.now().time()
                self.log.log(f"loop(): started: {t}")
                self.batch = self.batcher.getNextBatch()
                if self.batch is None:
                    # self.executor.submit(self.displayResults)
                    self.displayResults()
                    return
                self.executor.submit(self.process) # run on seperate thread
                # self.process() # run on same thread
                t = datetime.datetime.now().time()
                self.log.log(f"loop(): ended: {t}")
        except KeyboardInterrupt:
            self.executor.shutdown()
            self.log.log("displaying results")
            self.displayResults()
            self.log.log(f"Runner done()")
    
    def process(self):
        if self.batch is None:
            return
        t = datetime.datetime.now().time()
        self.log.log(f"process(): started: {t}")
        thisRes = self.hrEstimator.estimateHR(self.batch.copy())
        self.batch = None
        if thisRes == None:
            thisHr, thisSelectedHr = [-1, -1, -1], -1
        else:
            thisHr, thisSelectedHr = thisRes
        self.hr.append(thisHr)
        self.selectedHr.append(thisSelectedHr)
        self.display.render(self.camera.getLastReadFrame(), thisHr, thisSelectedHr, self.camera.getFrameNum()/self.camera.getFrameCount()*100)
        t = datetime.datetime.now().time()
        self.log.log(f"process(): ended: {t}")

    def displayResults(self):
        # convert hr list to np array
        self.hr = np.array(self.hr)
        
        fig = plt.figure()
        plt.suptitle("Heart Rate")
        plt.subplot(4, 1, 1)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.batcher.getSamplingRate(), self.hr.shape[0]), self.hr[:,0], 'blue', label='Channel 1')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.batcher.getSamplingRate(), self.hr.shape[0]), self.hr[:,1], 'green', label='Channel 2')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.batcher.getSamplingRate(), self.hr.shape[0]), self.hr[:,2], 'red', label='Channel 3')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(np.linspace(0, self.camera.getFrameCount()/self.batcher.getSamplingRate(), self.hr.shape[0]), self.selectedHr, 'black', label='Selected HR')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.legend()
        plt.show()
