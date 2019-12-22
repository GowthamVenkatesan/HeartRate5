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
    # def __init__(self, N = 2, fc = 5, fs = 30, debug=False):
        self.debug = debug
        self.window = window
        # self.filter_b, self.filter_a = signal.iirfilter(N, fc/(fs/2), btype='lowpass', analog=False)
        self.filter_b, self.filter_a = signal.butter(N, fc/(fs/2), analog=False)
        self.zi = None
        self.fs = fs
        self.log = Log('LowPassFilter')

        figure = plt.figure("LowPassFilter")
        plt.clf()

        self.channel = 0
        self.colors = [ "r", "g", "b" ]

        if False: #self.debug:
            self.log.log(f'Created iir lpf, N={N}, fc={fc}, fs={fs}, b={self.filter_b}, a={self.filter_a}')
            w, h = signal.freqz(self.filter_b, self.filter_a)
            
            fig = plt.figure("LowPassFilter")
            plt.clf()

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
            figure = plt.figure("LowPassFilter")

            plt.subplot(2,1,1)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), x, self.colors[self.channel])
            plt.title("raw")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.subplot(2,1,2)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), y, self.colors[self.channel])
            plt.title("filtered")
            plt.xlabel("time")
            plt.ylabel("signal")
            
            plt.draw()
            plt.pause(0.001)

            self.channel += 1
        return y

class BandPassFilter:

    def __init__(self, fs, N = 2, debug=False):
        self.N = N
        self.fs = fs
        self.debug = debug

        self.epsillon = 0.01
        self.log = Log("BandPassFilter")

        figure = plt.figure("BandPassFilter")
        plt.clf()

        self.channel = 0
        self.colors = ["r", "g", "b"]
    
    def filterSignal(self, x, fl, fh):
        fln = fl/(self.fs/2)
        fhn = fh/(self.fs/2)
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
            
        self.b, self.a = signal.iirfilter(self.N, [fln, fhn], btype='bandpass', analog=False)
        zi = signal.lfilter_zi(self.b, self.a)
        y, _ = signal.lfilter(self.b, self.a, x, zi=zi)

        if self.debug:
            # w, h = signal.freqz(self.b, self.a)
            # fig = plt.figure("BandPassFilter")
            # plt.title('IIR BPF Frequency Response')
            # ax1 = fig.add_subplot(111)
            # plt.plot(w, 20 * np.log10(abs(h)), 'b')
            # plt.ylabel('Amplitude [dB]', color='b')
            # plt.xlabel('Frequency [rad/sample]')
            # ax2 = ax1.twinx()
            # angles = np.unwrap(np.angle(h))
            # plt.plot(w, angles, 'g')
            # plt.grid()
            # plt.axis('tight')
            # plt.show()

            figure = plt.figure("BandPassFilter")
            
            plt.subplot(2,1,1)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), x, self.colors[self.channel])
            plt.title("raw")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.subplot(2,1,2)
            plt.plot(np.linspace(0, x.shape[0]/self.fs, x.shape[0]), y, self.colors[self.channel])
            plt.title("filtered")
            plt.xlabel("time")
            plt.ylabel("signal")
            
            plt.draw()
            plt.pause(0.001)

            self.channel += 1
        return y
