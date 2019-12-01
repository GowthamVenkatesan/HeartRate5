import numpy as np
from matplotlib import pyplot as plt

# 1000hz sine @ 100 samples a second
T = 1/100
fs = int(1/T)
f = 1000
length = 10
t = np.linspace(0, length, length*fs)
x = np.sin(2*np.pi*f*t)
y = np.fft.fft(x, n=10*fs)
freq = np.fft.fftfreq(length*fs, d=T)
yMax = y.max()
fMax = freq[ np.argmax(y) ]
print(f"yMax: {yMax}, fMax: {fMax}")
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("time domain")
plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(y))
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.title("frequency domain")
plt.show()
