
# coding: utf-8

import numpy as np
from numpy import array, where, nan, diff
from project_utilities import parse_rinex_obs_file
from types import SimpleNamespace
from datetime import datetime, timedelta
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.rcParams.update({'font.size': 10})


# download the appropriate RINEX file and use `parse_rinex_obs_file` to load it in
header, rinex = parse_rinex_obs_file("./nist2330.17o")
G15 = rinex['G15']
# this is the day for which we want an observation file
day_start_utc = datetime(2017, 8, 21, 0, 0, 0)
# this converts the datetime object into a single number that is in the format used 
# to store the time in the output of the `parse_rinex_obs_file` function
day_start_dt64 = np.datetime64(day_start_utc)

time = G15.time
G15L1 = G15.signals['L1']
G15L2 = G15.signals['L2']

diff = diff(G15L1.carrier)/30.0  # Dividing by 30 since the difference between each measurement is 30 seconds
diff = np.append(diff, nan)

G15DataFrame = pd.DataFrame({"pr": G15L1.pr, "L1 SNR": G15L1.snr, "L2 SNR": G15L2.snr, "L1 Doppler": G15L1.doppler,
                             "L2 Doppler": G15L2.doppler, "L1 Doppler Derived": diff}, index=time)


def plot_stuff():
    fig = plt.figure(figsize=(17,5))
    fig.subplots_adjust(wspace=.5)
    fig.suptitle('RINEX Data for PRN 15')
    x = G15DataFrame.index

    # PSUEDORANGE PLOT
    ax = fig.add_subplot(1, 4, 1)
    y = G15DataFrame['pr'].data
    ax.set_title('Psuedorange')
    ax.set_xlabel('Time (HH:MM:SS)')
    ax.set_ylabel('Psuedorange(m)')
    ax.set_xlim(x[0], x[-1])
    maxx = np.argmax(y)
    minx = np.argmin(y)
    ax.plot(x, y,'o', markersize=3)
    ax.plot(x[maxx], y[maxx], 'ro', markersize=6)
    ax.plot(x[minx], y[minx], 'ro', markersize=6)
    ax.text(x[minx] - timedelta(hours=10), y[minx] + 1e5, 'Min:\n' + x[minx].strftime('%H:%M:%S') + ',\n' +
            str(y[minx]))
    ax.text(x[maxx], y[maxx] - 7e5, 'Max:\n' + x[maxx].strftime('%H:%M:%S') + ',\n' + str(y[maxx]))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # SNR PLOT
    ax = fig.add_subplot(1, 4, 2)
    y1 = G15DataFrame['L1 SNR']
    y2 = G15DataFrame['L2 SNR']
    ax.set_title('SNR')
    ax.set_xlabel('Time (HH:MM:SS)')
    ax.set_ylabel('Signal-to-Noise Ratio (dB)')
    ax.set_xlim(x[0], x[-1])
    ax.plot(x, y1, 'o', x, y2, 'o')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.legend(['L1', 'L2'])

    # DOPPLER PLOT
    ax = fig.add_subplot(1, 4, 3)
    y1 = G15DataFrame['L1 Doppler']
    y2 = G15DataFrame['L2 Doppler']
    ax.set_title('L1 Doppler Frequency')
    ax.set_xlabel('Time (HH:MM:SS)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(x[0], x[-1])
    ax.plot(x, y1, 'o', x, y2, 'o')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.legend(['L1', 'L2'])

    # DERIVED DOPPLER PLOT
    ax = fig.add_subplot(1, 4, 4)
    # y1 = G15DataFrame['L1 Doppler']
    y2 = G15DataFrame['L1 Doppler Derived']
    ax.set_title('L1 Doppler Derived')
    ax.set_xlabel('Time (HH:MM:SS)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-5000, 5000)
    ax.plot(x, y1, 'o', x, y2, 'o')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.legend(['L1', 'L1 d/dx'])

    fig.autofmt_xdate()
    plt.show()

plot_stuff()
