import time

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
# from sklearn.preprocessing import normalize

from ..util.log import Log

class IndependentComponentAnalysis:

    def __init__(self, debug=True):
        self.log = Log("IndependentComponentAnalysis")
        self.debug = debug
        self.firstRun = True

        self.ica = FastICA(n_components=3, max_iter=200)
    
    def fitTransform(self, X):
        # if self.firstRun:
        #     self.firstRun = False
        #     return self.ica.fit_transform(X)
        # else:
        #     return self.ica.transform(X)
        Y = self.ica.fit_transform(X)
        # self.log.log(f"X.shape: {X.shape}")
        # self.log.log(f"Y.shape: {Y.shape}")
        if self.debug:
            plt.figure("IndependentComponentAnalysis")
            plt.clf()
            
            plt.subplot(2,1,1)
            plt.plot(X)
            plt.subplot(2,1,2)
            plt.plot(Y)

            plt.draw()
            plt.pause(0.001)
        return Y

    def getMixingMartrix(self):
        return self.ica.mixing_

class LowPassFilter:

    def __init__(self, fs, N = 7, fc = 5, debug=False, window=True):
        self.debug = debug
        self.window = window
        self.fs = fs
        self.fc = fc
        self.N = N
        self.log = Log('LowPassFilter')

    def filterSignal(self, x):
        y = self.butter_lowpass_filter(x, self.fc, self.fs, self.N)
        return y
    
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=7):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        self.zi = signal.lfilter_zi(b, a)
        y = signal.lfilter(b, a, data)
        return y

class BandPassFilter:

    def __init__(self, fs, N = 2, debug=False):
        self.N = N
        self.fs = fs
        self.debug = debug

        self.epsillon = 0.01
        self.log = Log("BandPassFilter")
    
    def filterSignal(self, x, fl, fh):
        nyq = 0.5 * self.fs
        fln = fl / nyq
        fhn = fh / nyq
        if fln <= 0:
            fln = 0 + self.epsillon
            self.log.log(f"saturated fln: {fln}, original fl: {fl}")
        elif fln >= 1:
            fln = 1 - self.epsillon
            self.log.log(f"saturated fln: {fln}, original fl: {fl}")
        if fhn <= 0:
            fhn = 0 + self.epsillon
            self.log.log(f"saturated fhn: {fhn}, original fh: {fh}")
        elif fhn >= 1:
            fhn = 1 - self.epsillon
            self.log.log(f"saturated fhn: {fhn}, original fh: {fh}")
        
        y = self.butter_bandpass_filter(x, fln, fhn, self.fs, order=self.N)
        return y

    def butter_bandpass(self, fs, fl, fh, order=2):
        b, a = signal.butter(order, [fl, fh], btype='band', analog=False)
        return b, a

    def butter_bandpass_filter(self, data, fl, fh, fs, order=7):
        b, a = self.butter_bandpass(fs, fl, fh, order=order)
        self.zi = signal.lfilter_zi(b, a)
        y = signal.lfilter(b, a, data)
        return y