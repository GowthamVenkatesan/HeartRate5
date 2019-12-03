# we don't care

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..lib.signal_processor import LowPassFilter, BandPassFilter, Runner
from ..util.log import Log

log = Log('signal_processor_test')

t = np.linspace(0, 2, 61)
x1 = np.sin(2*np.pi*3*t)
x2 = np.sin(2*np.pi*20*t)
x = x1 + x2

log.log('Performing lpf test')
lpf = LowPassFilter(debug=True)
xFilt = lpf.filterSignal(x)
fig = plt.figure()
plt.plot(t, x1, 'b')
plt.plot(t, x2, 'r')
plt.plot(t, x, 'black')
plt.plot(t, xFilt, 'g')
plt.show()

log.log('Preforming bpf test')
bpf = BandPassFilter(debug=True)
xFilt = bpf.filterSignal(x, 1, 5)
fig = plt.figure()
plt.plot(t, x1, 'b')
plt.plot(t, x2, 'r')
plt.plot(t, x, 'black')
plt.plot(t, xFilt, 'g')
plt.show()
