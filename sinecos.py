import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi,np.pi,1000)
plt.plot(x,np.sin(x)+np.cos(x))
plt.xlabel('Angle in radians')
plt.ylabel('Sinucoisal wave')
plt.title("Combined waves")
plt.grid(True)
plt.axis("tight")
plt.show()
