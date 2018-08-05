import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
dt = 0.01 #Sampling time period
Fs = 1/dt #Sampling frequency
t = np.arange(0,10,dt)

nse = np.random.randn(len(t))
r = np.exp (-t/0.05)
cnse = np.convolve(nse,r)*dt
cnse = cnse[:len(t)]

sig1 = signal.sawtooth(2*np.pi*t)#+cnse #The sawtooth signal


fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(9,9))

axes[0,0].plot(t,sig1,color='C0')
axes[0,0].grid(True,color='black')
axes[0,0].set_xlabel("Time interval")
axes[0,0].set_ylabel("Sawtooth signal with error")
axes[0,0].set_title("The Main Signal")

axes[0,1].magnitude_spectrum(sig1,Fs=Fs,color='C1')
axes[0,1].grid(True,color='black')
axes[0,1].set_title("Magnitude spectrum")

axes[1,0].phase_spectrum(sig1,Fs=Fs,color='C2')
axes[1,0].grid(True,color='black')
axes[1,0].set_title("Phase spectrum")

axes[1,1].angle_spectrum(sig1,Fs=Fs,color='C3')
axes[1,1].grid(True,color='black')
axes[1,1].set_title("Angle Spectrum")

#plt.show()


Fs2 = 10000
dt2 = 1/10000
t2 = np.arange(0,1,dt2)

nse2 = np.random.randn(len(t2))
r2=np.exp(-t2/0.05)
cnse2 = np.convolve(nse2,r2)*dt2
cnse2=cnse2[:len(t2)]

sig2 = signal.square(2*np.pi*1100*t2)+cnse2

fig2,axes = plt.subplots(nrows=2,ncols=2,figsize=(9,9))

axes[0,0].plot(t2,sig2,color='C0')
axes[0,0].grid(True,color='black')
axes[0,0].set_xlabel("Time interval")
axes[0,0].set_ylabel("Square wave with error")
axes[0,0].set_title("The Main Signal")

axes[0,1].magnitude_spectrum(sig2,Fs=Fs2,color='C1')
axes[0,1].grid(True,color='black')
axes[0,1].set_title("Magnitude spectrum")

axes[1,0].phase_spectrum(sig2,Fs=Fs2,color='C2')
axes[1,0].grid(True,color='black')
axes[1,0].set_title("Phase spectrum")

axes[1,1].angle_spectrum(sig2,Fs=Fs2,color='C3')
axes[1,1].grid(True,color='black')
axes[1,1].set_title("Angle Spectrum")

plt.show()




