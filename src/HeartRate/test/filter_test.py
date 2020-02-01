import numpy as np
import scipy
import matplotlib.pyplot as plt

from ..lib.signal_processor import LowPassFilter, BandPassFilter
from ..util.log import Log

log = Log("FilterTest")

log.log(f"LowPassFilterTest")
N = 7
fs = 30
fc = 5
t = np.linspace(0, 2, fs*2)
x1 = np.sin(2*np.pi*3*t)
x2 = 2*np.sin(2*np.pi*8*t)
x = x1 + x2
lpf = LowPassFilter(fs, N, fc, debug=False)
y = lpf.filterSignal(x)
plt.subplot(4, 1, 1)
plt.plot(t, x1, 'r')
plt.subplot(4, 1, 2)
plt.plot(t, x2, 'r')
plt.subplot(4, 1, 3)
plt.plot(t, x, 'b')
plt.subplot(4, 1, 4)
plt.plot(t, y, 'g')
plt.show()


log.log(f"BandPassFilterTest")
N = 2
fs = 30
bpf = BandPassFilter(fs, N, debug=False)
t = np.linspace(0, 1, fs*2)
x1 = 1*np.sin(2*np.pi*2*t)
x2 = np.sin(2*np.pi*3*t)
x3 = 1*np.sin(2*np.pi*4*t)
x = x1 + x2 + x3
bpf = BandPassFilter(fs, N=2, debug=False)
y = bpf.filterSignal(x, 0.8*3, 1.2*3)
plt.subplot(5, 1, 1)
plt.plot(t, x1, 'r')
plt.subplot(5, 1, 2)
plt.plot(t, x2, 'r')
plt.subplot(5, 1, 3)
plt.plot(t, x3, 'r')
plt.subplot(5, 1, 4)
plt.plot(t, x, 'b')
plt.subplot(5, 1, 5)
plt.plot(t, y, 'g')
plt.show()


log.log(f"done")