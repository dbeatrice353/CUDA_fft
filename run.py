from matplotlib import pyplot as plt
import numpy as np
import os

t = np.linspace(0, 10*np.pi, 256)
waveform = np.sin(2*t) + np.sin(3*t) + np.sin(5*t)
np.savetxt("input.txt",waveform,fmt="%5.10f")
os.system("fft1d.exe")
result = np.loadtxt("output.txt",delimiter="\t",usecols=[0])
"""
plt.figure(1)
plt.plot(waveform)
plt.figure(2)
plt.plot(result)
plt.show()
"""
