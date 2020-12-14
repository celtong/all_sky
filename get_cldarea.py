from datetime import datetime
from datetime import timedelta
from datetime import timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import imageio
import pysolar
from defisheye import Defisheye
from pytz import timezone

# Approximation of the sun area during a day
sun = pd.read_csv('/Users/celestie_/Desktop/clouds/sun.txt')        #the file of sun area
index = sun['index']
sunarea = sun['sun']
qua = (abs(index-390))**2*1.2+72000       # quadratic approach
# sin = np.cos((index-390)/640*np.pi-np.pi)*125000+190000     #sin approach (not good)
plt.plot(index, sunarea)
plt.plot(index, qua, label='(abs(index-390))^2*1.2+72000')
plt.legend()
plt.title('sun area in a day 10.07')
plt.ylabel('Number of Pixels of the Sun')
plt.xlabel('Index from the First Image')
plt.show()


def sunmask():
    '''
    to write the adjusted cloud fraction into a file
    '''
    file = open("/Users/celestie_/Desktop/1005.txt", 'w')
    file.write("index,UTC_TIME,adjusted_cldfraction\n")
    # to loop over every image on that day
    for i in range(42, 416):
        date = datetime(2020, 10, 5, 15, 48, tzinfo=timezone('UTC')) + timedelta(minutes=i)
        # the quadratic equation to estimate sun area based on time / index
        sun = (abs(i-400))**2*1.2 + 72000
        # load image
        img = imageio.imread('/Users/celestie_/Desktop/defish1005_1009/1005_' + str(i) + '.jpg')
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        # index of pixels that are classified as clouds
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        # fct = cloudx.size * 100 / 392639      #original method to get cloud fraction
        # the adjusted cloud fraction
        nfct = (cloudx.size-sun) * 100 / (392639-sun)
        # fix invalids
        if nfct < 0:
            nfct = 0
        if nfct > 100:
            nfct = 100
        file.write(str(i)+','+str(date)+','+str(nfct) + '\n')
    file.close()


# sunmask()

'''
some messy codes to visualize the computer vision (cloud/no cloud, sun/not sun, etc) 
'''
# img = imageio.imread('/Users/celestie_/Desktop/defish/1018_490.jpg')
# sunx, suny = np.where((img[:, :, 0]/img[:, :, 2] > 0.4)&(img[:, :, 2]>120))
# sun = np.zeros([1072, 1072])
# sun[sunx, suny] = 1
# print(sunx.size)      # higher sun 83559
# rbr = plt.contourf(sun, cmap='bone')
# ratio = img[:, :, 0] / img[:, :, 2]
# mask = np.zeros((1072, 1072))
# vx, vy = np.where((ratio>5) & (ratio < 150))
# cloudx, cloudy = np.where((ratio>=0.4) & (img[:, :, 2]>=120))
# mask[vx, vy] = ratio[vx, vy]
# mask[cloudx, cloudy] = 1
# sun = (abs(490-400))**2*1.2 + 72000
# nfct = (cloudx.size-sun) * 100 / (392639-sun)
# print(nfct)
# rbr = plt

'''
# trying out visualizing the data
f = pd.read_csv('TOA5_SonicMeans_2020_08_28_2221.dat', skiprows=[2,3],
                header=1,na_values='NAN', parse_dates=True)
radf = pd.read_csv('TOA_NetRad_2020_08_28.dat', skiprows=[2,3],
                header=1,na_values='NAN', parse_dates=True)
tf = pd.read_csv('TempRH_2020_08_28_2220.dat', skiprows=[2,3],
                header=1,na_values='NAN', parse_dates=True)
t = tf['TIMESTAMP']
time = pd.to_datetime(t)
ts = f["Ts_Avg"]
tsk = ts+273
swin = radf['Incoming_SW']
swout = radf['Outgoing_SW']
lwout = radf['Outgoing_LW']
lwin = radf['Incoming_LW']
temp2 = tf['ST110_2m_C']
temp10 = tf['ST110_10m_C']
fig, ax = plt.subplots()
# ax.scatter(tsk, lwout, s=0.5, c='black')
# ax.set_xlabel('temp K')
# ax.set_ylabel('OLR W m-2')
ax.plot(time, temp2, label='2m', c='b')
ax.plot(time, temp10, label='10m', c='r')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))       #separate by every 12h
plt.xticks(rotation=45)
ax.legend(loc=1)
ax.set_ylabel('temp degree C')
# ax1 = ax.twinx()
# ax1.plot(time, ts, label='sonic temp', c='r')
# ax1.set_ylabel('temp degree C')
# ax1.legend(loc=2)
plt.show()
'''

lat = 38.19
lon = -78.69



def getCFA():
    '''
    to get the cloud fraction with the original method from 1005-1009
    '''
    CFAfile = open("/Users/celestie_/Desktop/cfa.txt", 'w')
    # CFAfile.write('10.05 cloud fraction\n')
    # to loop through every image on 10.5
    for i in range(11, 416):
        img = imageio.imread('/Users/celestie_/Desktop/defish/1005_'+str(i)+'.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size*100 / 392639
        CFAfile.write(str(fct)+'\n')
    # manually set 10.6 to clear (because of sun detection failure)
    for i in range(0, 1221+623):
        CFAfile.write('0\n')
    for i in range(0, 280-43):
        img = imageio.imread('/Users/celestie_/Desktop/defish/1009_' + str(i+43) + '.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size * 100 / 392639
        CFAfile.write(str(fct) + '\n')
    for i in range(284, 636):
        img = imageio.imread('/Users/celestie_/Desktop/defish/1009_' + str(i) + '.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size * 100 / 392639
        CFAfile.write(str(fct) + '\n')
    CFAfile.close()


# img = imageio.imread('/Users/celestie_/Desktop/defish/1005_378.jpg')
# # print(color.shape)
# ratio = img[:, :, 0] / img[:, :, 2]    #red/blue
# # plt.plot(ratio[:, 800])
# # plt.show()
# cloudx, cloudy = np.where((ratio>=0.4) & (img[:, :, 2]>=120))
# # fct = cloudx.size*100 / 392639
# # print(fct)
# mask = np.zeros((1072, 1072))
# mask[cloudx, cloudy] = 1 #color[a, b, 0]
# rbr = plt.contourf(mask, cmap='bone')
# plt.gca().invert_yaxis()
# # plt.colorbar(rbr, label='blue')
# plt.title('cloud')
# plt.show()
# r = plt.contourf(ratio, cmap='bone')
# plt.gca().invert_yaxis()
# plt.colorbar(r, label='red')
# plt.show()


# getCFA()

def solarin(list):
    '''
    list: incoming solar on a certain day
    '''
    for i in range(0, 405):
        date = datetime(2020, 10, 5)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        list.append(pysolar.radiation.get_radiation_direct(date, alt))

def rtDate():
    '''
    to write date, cloud fraction, and solar_in into one file
    '''
    file = open("/Users/celestie_/Desktop/CFA1016_102.csv", 'w')
    # csv_writer = writer(file)
    for i in range(0, 640):
        date = datetime(2020, 10, 16, 12, 30, tzinfo=timezone('UTC'))+timedelta(minutes=i)
        img = imageio.imread('/Users/celestie_/Desktop/defish/1005_' + str(i) + '.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size * 100 / 392639
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date)+', '+str(fct) + ', '+ str(solar_in)+'\n')
    for i in range(0, 640):
        date = datetime(2020, 10, 17, 12, 30, tzinfo=timezone('UTC')) + timedelta(minutes=i)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date) + ', 0, '+ str(solar_in)+'\n')
    for i in range(0, 640):
        date = datetime(2020, 10, 18, 12, 30, tzinfo=timezone('UTC')) + timedelta(minutes=i)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date) + ', 0, ' + str(solar_in) + '\n')
    for i in range(0, 640):
        date = datetime(2020, 10, 19, 12, 30, tzinfo=timezone('UTC')) + timedelta(minutes=i)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date) + ', 0, ' + str(solar_in) + '\n')
    for i in range(0, 640):
        img = imageio.imread('/Users/celestie_/Desktop/defish/1009_' + str(i) + '.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size * 100 / 392639
        date = datetime(2020, 10, 20, 12, 30, tzinfo=timezone('UTC')) + timedelta(minutes=i)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date) + ', '+str(fct) + ', ' + str(solar_in)+'\n')
    for i in range(284, 636):
        img = imageio.imread('/Users/celestie_/Desktop/defish/1009_' + str(i) + '.jpg')
        # print(color.shape)
        ratio = img[:, :, 0] / img[:, :, 2]  # red/blue
        cloudx, cloudy = np.where((ratio >= 0.4) & (img[:, :, 2] >= 120))
        fct = cloudx.size * 100 / 392639
        date = datetime(2020, 10, 9, 17, 31, tzinfo=timezone('UTC')) + timedelta(minutes=i-284)
        alt = pysolar.solar.get_altitude(lat, lon, date)
        solar_in = pysolar.radiation.get_radiation_direct(date, alt)
        file.write(str(date) + ', '+str(fct) + ', ' + str(solar_in)+'\n')
    file.close()

# rtDate()

def combine():
    radf = pd.read_csv('100920.csv',
                header=0,na_values='NAN', parse_dates=True)
    cldf = pd.read_csv('output.csv', header=0)
    merged = pd.merge(radf, cldf, "right")
    merged.to_csv("output1.csv", index=False)

# combine()

def skyMask():
    '''
    to get the sky area
    '''
    color = imageio.imread('/Users/celestie_/Desktop/defish/1005_369.jpg')  #clear sky
    a, b = np.where((color[:, :, 2] >= 120))        # sky
    mask = np.zeros((1072, 1072))
    mask[a, b] = 1 #color[a, b, 0]
    print(np.sum(mask))        # number of pixels of the sky 392639

    rbr = plt.contourf(mask, cmap='bone')
    plt.gca().invert_yaxis()
    # plt.colorbar(rbr, label='blue')
    plt.title('non-sky mask B>=120')
    plt.show()

# skyMask()

def defish(n, date=""):
    '''
    defisheye the images
    n: number of photos
    date: MMDD
    '''
    dtype = 'orthographic'
    format = 'circular'
    fov = 180  # field of view
    pfov = 120  # perspective fov, default 120
    # dir = "/Users/celestie_/Desktop/skyimg"
    # newdir = "/Users/celestie_/Desktop/defish"
    # img = "/Users/celestie_/Desktop/0928.png"
    # img_out = f"/Users/celestie_/Desktop/0928_{dtype}_{format}_{pfov}_{fov}.png"
    for i in range(1, n+1):
        if (i < 10):
            # img = "/Users/celestie_/Desktop/skyimg/ezgif-frame-00"+str(i)+".jpg"
            img = "/Users/celestie_/Desktop/"+date+"/00" + str(i) + ".bmp"
        elif (i >= 100):
            img = "/Users/celestie_/Desktop/"+date+"/" + str(i) + ".bmp"
        else:
            img = "/Users/celestie_/Desktop/"+date+"/0" + str(i) + ".bmp"
        img_out = f"/Users/celestie_/Desktop/defish/"+date+"_" + str(i) + ".jpg"
        obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
        obj.convert(img_out)


# defish(640, '1026')
# defish(381, '1016')


# cb("CFA1005_1009.txt", "cfa.txt")