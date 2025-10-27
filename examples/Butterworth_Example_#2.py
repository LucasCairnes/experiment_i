## Butterworth Filter Example 2 ##
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

Time = 1.0
Samples = 1000

f1 = 10
f2 = 30
noise = np.random.normal(0,1,Samples)

t = np.linspace(0, Time, Samples, False) 

#original signal (f1,  f2) + noise
sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + noise

#parameters for the filter
Rate =  Samples/Time
Low_f = 15
High_f = 45
Order = 5

LP = signal.butter(Order, [Low_f], 'Lp', fs = Rate, output='sos') #lowpass
HP = signal.butter(Order, [High_f], 'Hp', fs = Rate, output='sos') #highpass
BP = signal.butter(Order, [Low_f,High_f], 'Bp', fs = Rate, output='sos') #bandpass


#apply filters
filtered_LP = signal.sosfilt(LP, sig)
filtered_HP = signal.sosfilt(HP, sig)
filtered_BP = signal.sosfilt(BP, sig)

#create filter profiles
w_LP,h_LP = signal.sosfreqz(LP,fs = Rate)
w_BP,h_BP = signal.sosfreqz(BP,fs = Rate)
w_HP,h_HP = signal.sosfreqz(HP,fs = Rate)

#plot results
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=3, ncols=2)

fig.suptitle( "Butterworth Lowpass: n=" + str (Order) + ", f$_{L}$ = "  + str(Low_f) + "Hz\n" +
              "Butterworth Highpass: n=" + str (Order) + ", f$_{H}$ = "  + str(High_f) + "Hz\n"
              "Butterworth Bandpass: n=" + str (Order) + ", f$_{L}$ = "  + str(Low_f) +
              "Hz" + ", f$_{H}$ = "  + str(High_f) + "Hz",            
             fontsize=12, x = 0.04, y = 0.95, ha = 'left')

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, sig,'lightgrey')
ax0.plot(t,filtered_LP, 'r')
ax0.set_xlim(0,1)
ax0.set_ylim(-5,5)
ax0.set_xlabel("time (s)")
ax0.legend(['signal + noise','Lowpass filter'],bbox_to_anchor=(1.0,1.0))

ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(t, sig,'lightgrey')
ax1.plot(t, filtered_HP, 'b')
ax1.set_xlim(0,1)
ax1.set_ylim(-5,5)
ax1.set_xlabel("time (s)")
ax1.legend(['signal + noise','Highpass filter'],bbox_to_anchor=(1.0,1.0))

ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(t, sig,'lightgrey')
ax2.plot(t, filtered_BP, 'g')
ax2.set_xlim(0,1)
ax2.set_ylim(-5,5)
ax2.set_xlabel("time (s)")
ax2.legend(['signal + noise','Bandpass'],bbox_to_anchor=(1.0,1.0))

ax3 = fig.add_subplot(gs[:, 1])
ax3.semilogx(w_LP,abs(h_LP),'r',label = 'Lowpass')
ax3.semilogx(w_BP,abs(h_BP),'b',label = 'Bandpass')
ax3.semilogx(w_HP,abs(h_HP),'g',label = 'Highpass')
ax3.axvline(Low_f, color ='grey', label = 'f$_{L}$,f$_{H}$')
ax3.axvline(High_f, color ='grey')
ax3.axhline(1/np.sqrt(2), color = 'black', linestyle = '--', label = '$\dfrac{1}{\sqrt{2}}$')
ax3.set_xlim(0.1*Low_f,10*High_f)
ax3.set_ylim(0,1.1)
ax3.set_xlabel("log(f)")
ax3.legend(bbox_to_anchor=(1.0,0.6))

plt.tight_layout()
plt.show()