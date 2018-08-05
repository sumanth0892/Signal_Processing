import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi,np.pi,200)
sig1 = np.sin(x)
sig2 = np.cos(x)
fig,axes = plt.subplots(nrows = 2,ncols = 2,figsize = (9,9))

axes[0,0].set_title("Sine wave")
axes[0,0].plot(x,sig1,color = 'C0')
axes[0,0].set_xlabel('Angle in Radians')
axes[0,0].set_ylabel('Sine of X')
axes[0,0].grid(True,color='black')

axes[0,1].set_title("Cosine wave")
axes[0,1].plot(x,sig2,color = 'C1')
axes[0,1].set_xlabel('Angle in Radians')
axes[0,1].set_ylabel('Cosine of X')
axes[0,1].grid(True,color='black')

axes[1,0].set_title("Additive wave")
axes[1,0].plot(x,sig1,color = 'C2')
axes[1,0].set_xlabel('Angle in Radians')
axes[1,0].set_ylabel('Sine + Cosine of X')
axes[1,0].grid(True,color='black')

axes[1,1].set_title("Subtractive wave")
axes[1,1].plot(x,sig1,color = 'C3')
axes[1,1].set_xlabel('Angle in Radians')
axes[1,1].set_ylabel('Sine of X')
axes[1,1].grid(True,color='black')

plt.show()
