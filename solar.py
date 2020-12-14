import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pysolar
import pytz

import functions
import matplotlib.dates as mdates

lat = 38.19
lon = -78.69

f = pd.read_csv('/Users/celestie_/Documents/academic/research/py/TOA_NetRad_2020_09_25.dat', header=1,skiprows=[2,3])

D = 6
M = 10
Y = 20
J = D - 32 + 275//9*M + 2*3//(M+1)+int(M/100-(Y%4)/4+0.975)
dr = 1+0.033*np.cos(2*np.pi/365*J)
delta = 0.409*np.sin(2*np.pi/365*J - 1.39)
phi = lat*np.pi/180
Gsc = 4.29  #solar constant
t1 = 1/60      # 1-hour frequency
t =  np.arange(4, 20+1/60, 1/60)
Lz = 75     # center longitude of the timezone
Lm = 78.69      #longitude
b = 2*np.pi*(J-81)/364
Sc = 0.1645 * np.sin(2*b) - 0.1255*np.cos(b) - 0.025*np.sin(b)
omega = np.pi/12*((t+0.06667*(Lz-Lm)+Sc)-12)
omegas = np.arccos((-np.tan(phi)*np.tan(delta)))
omega1 = omega - np.pi*t1/24
omega2 = omega + np.pi*t1/24
a = (np.where(omega1<-omegas))[0]
b = (np.where(omega1>omegas))[0]
c = (np.where(omega2<-omegas))[0]
d = (np.where(omega2>omegas))[0]
e = (np.where(omega1>omega2))[0]
omega1[a] = -omegas
omega1[b] = omegas
omega2[c] = -omegas
omega2[d] = omegas
omega1[e] = omega2[e]
Ra = 12/np.pi*Gsc*dr*((omega2-omega1)*np.sin(phi)*np.sin(delta)+
                      np.cos(phi)*np.cos(delta)*(np.sin(omega2)-np.sin(omega1)))
z=300
Rso = Ra*(0.75+2E-5*z)
RsoWm = Rso * 1E6 /60
# RsoWm = Rso * 277.778 * 60      # MJ to W*minute


ts = f['TIMESTAMP']
swin = f["Incoming_SW"]
t = pd.to_datetime(ts)
cmp = (np.where((t>=datetime(2020, 10, 6, 9, 00))&(t<=datetime(2020, 10, 7, 1, 00))))[0]
time = t[cmp]
sfcsw = swin[cmp]


fig, ax = plt.subplots()
ax.plot(time, RsoWm, c='black', label='calc minutely')
# ax1 = ax.twinx()
ax.plot(time, swin[cmp], label='measurements')
ax.set_ylabel('W m-2')
# ax1.set_ylabel('meas W m-2')
plt.title('Comparison of surface sw_in 10.06')
plt.legend()
plt.show()