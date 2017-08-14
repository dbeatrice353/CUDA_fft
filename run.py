from matplotlib import pyplot as plt
import numpy as np
import os

# 256 samples
N = 256

# create a waveform and save it to a file
t = np.linspace(0, 10*np.pi, N)
signal = np.zeros((N,2))
signal[:,0] = np.sin(2*t) + np.sin(3*t) + np.sin(5*t)
np.savetxt("input.txt",signal,fmt="%5.10f",delimiter="\t")

# compile the CUDA/c file
os.system("make.bat")

# run
os.system("fft1d.exe")

# read in the spectrum data
result = np.loadtxt("output.txt",delimiter="\t",usecols=[0])

# plot
plt.figure(1)
plt.plot(signal)
plt.figure(2)
plt.plot(result)
plt.show()
