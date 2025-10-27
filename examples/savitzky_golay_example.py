## Savitzky-Golay filter example ##
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


Time = 1.0
Samples = 1000

f1 = 10
f2 = 20
noise = np.random.normal(0,1,Samples)

t = np.linspace(0, Time, Samples, False) 

#original signal (f1,  f2) + noise
sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + noise 


#filter paramters: window is number of coefficients, order is the polynomial order
window = 75
Order = 7

#apply filter
filtered = savgol_filter(sig, window_length = int(window), polyorder = int(Order), mode = 'interp')

#plot results
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=2, ncols=1)

fig.suptitle('Savitzky-Golay Filter\n' + 'Orignal signal: f$_{1}$= ' + str(f1) +
             'Hz f$_{2}$= ' + str(f2) + 'Hz', fontsize=12, x = 0.04, y = 0.95, ha = 'left')

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, sig, 'lightgrey')
ax0.plot(t,np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t),'b')
ax0.set_xlim(0,1)
ax0.set_ylim(-5,5)
ax0.set_xlabel("time (s)")
ax0.legend(['signal + noise','signal'],bbox_to_anchor=(1.0,1.0)) 

ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(t, sig, 'lightgrey')
ax1.plot(t,filtered,'r')
ax1.set_xlim(0,1)
ax1.set_ylim(-5,5)
ax1.set_xlabel("time (s)")
ax1.legend(['signal + noise','filtered signal'],bbox_to_anchor=(1.0,1.0)) 

plt.tight_layout()
plt.show()
