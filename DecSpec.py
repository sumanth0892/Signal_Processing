#Spectral representation
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
dt = 0.01 #Sampling interval
Fs = 1/dt #Sampling Frequency
t = np.arange(0,10,dt)

#generate noise
nse = np.random.randn(len(t))
r = np.exp(-t/0.05)
cnse = np.convolve(nse,r)*dt
cnse = cnse[:len(t)]

s = 0.1*np.sin(2*np.pi*t)+cnse #The signal is cut here since we cannot get a complete sine wave

fig,axes = plt.subplots(nrows = 2, ncols = 2, figsize=(7,7))

#Plot time signal
axes[0,0].set_title("Signal")
axes[0,0].plot(t,s,color = 'C0')
axes[0,0].set_xlabel("Time")
axes[0,0].set_ylabel('Amplitude')
axes[0,0].grid(True,color = 'black')

#Plot different spectra
axes[1,0].set_title("Magnitude Spectrum")
axes[1,0].magnitude_spectrum(s,Fs = Fs,scale = 'dB',color = 'C1')
axes[1,0].grid(True,color = 'black')

axes[0,1].set_title("Phase spectrum")
axes[0,1].phase_spectrum(s,Fs=Fs,color='C2')
axes[0,1].grid(True,color = 'black')

axes[1,1].set_title("Angle Spectrum")
axes[1,1].angle_spectrum(s,Fs=Fs,color = 'C3')
axes[1,1].grid(True,color='black')


fig.tight_layout()
plt.show()
