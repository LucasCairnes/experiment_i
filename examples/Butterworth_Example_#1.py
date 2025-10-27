## Butterworth Filter Example ##
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

Time = 1.0
Samples = 1000


f1 = 20
f2 = 35
noise = np.random.normal(0,1,Samples)

t = np.linspace(0, Time, Samples, False) 
sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + noise #original signal (f1,  f2) + noise

#parameters for the filter
Rate =  Samples/Time
Low_f = 10
High_f = 50
Order = 3

#create a butterworth bandpass filter 
sos = signal.butter(Order, [Low_f,High_f], 'bp', fs = Rate, output='sos') #Bandpass
filtered = signal.sosfilt(sos, sig) #apply filter


w,h = signal.sosfreqz(sos, fs = Rate) #create filter profile

# plot results
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=2, ncols=2)

fig.suptitle( "Butterworth Bandpass: n=" + str (Order) + ", f$_{1}$ = "  + str(Low_f) +
             "Hz, f$_{2}$ = "  + str(High_f) + "Hz", fontsize=16, x = 0.3, y = 0.95)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, sig,'lightgrey')
ax0.plot(t, np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t), 'r')
ax0.set_xlim(0,1)
ax0.set_ylim(-5,5)
ax0.set_xlabel("time (s)")
ax0.legend(['signal + noise','signal'],bbox_to_anchor=(1.0,1.0))

ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(t, sig,'lightgrey')
ax1.plot(t, filtered, 'b')
ax1.set_xlim(0,1)
ax1.set_ylim(-5,5)
ax1.set_xlabel("time (s)")
ax1.legend(['signal + noise','filtered result'],bbox_to_anchor=(1.0,1.0))

ax2 = fig.add_subplot(gs[:, 1])
ax2.semilogx(w,abs(h),'g',label = 'filter')
ax2.axvline(Low_f, color ='grey', label = 'f$_{1}$,f$_{2}$')
ax2.axvline(High_f, color ='grey')
ax2.axhline(1/np.sqrt(2), color = 'black', linestyle = '--', label = '$\dfrac{1}{\sqrt{2}}$')
ax2.set_xlim(0.1*Low_f,10*High_f)
ax2.set_ylim(0,1.1)
ax2.set_xlabel("log(f)")
ax2.legend(bbox_to_anchor=(1.0,1.0))


plt.tight_layout()
plt.show()


