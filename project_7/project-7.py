
# coding: utf-8

# # ASEN 5090 Assignment 7

# In[ ]:

import numpy as np
from numpy import array, datetime64, exp, where, diff, nan, concatenate, ndarray
from types import SimpleNamespace
from datetime import datetime, timezone, time
import scipy.constants as constants
import pandas as pd
from rinex_utilities import parse_rinex_obs_file
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from SatClass import Satellite

# CDDIS FTP path:  ftp://cddis.gsfc.nasa.gov/gnss/data/daily/2017/223/17o/
rinex_filename1 = './nist2320.17o'  # August 20th 2017
rinex_filename2 = './nist2330.17o'  # August 21st 2017
header1, observations1 = parse_rinex_obs_file(rinex_filename1)
header2, observations2 = parse_rinex_obs_file(rinex_filename2)

fL1 = 1.57542e9
fL2 = 1.2276e9
c = 299792458

kappa = (constants.elementary_charge ** 2)/(8 * (constants.pi ** 2) * constants.epsilon_0 * constants.electron_mass)
kappa = float("{0:.5g}".format(kappa))
print(kappa)

def plot_TECu_two_days(Satellite):
    time1, y1, time2, y2 = Satellite.get_two_day_plot_items()

    # Cleaning up these arrays since they wrap around and draw a line across
    # the entire plot
    time2 = np.delete(time2,-1)
    y2 = np.delete(y2, -1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(0, 24)
    line1, = ax.plot(time1, y1/1e16)
    ylim = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_xlim(0,24)
    ax2.set_ylim(ylim)
    line2, = ax2.plot(time2, y2/1e16, 'r')
    plt.legend([line1, line2], ['Aug. 20','Aug. 21'])
    plt.xlabel('Time (Hours)')
    plt.ylabel('TECu')
    plt.title('TECu for PRN {0}'.format(Satellite.prn))
    plt.show()

Sat = Satellite('G11', observations1, observations2)
plot_TECu_two_days(Sat)

def plot_question5(time, *satellites):
    for satellite in satellites:  # type: Satellite
        TEC16_1 = satellite.aug_20.ix[time]['TEC'].values[0]
        TEC16_2 = satellite.aug_21.ix[time]['TEC'].values[0]
        time1, y1, time2, y2 = satellite.get_two_day_plot_items()
        y1 = y1 - TEC16_1
        y2 = y2 - TEC16_2

        max1, max2 = np.max(np.abs(y1)), np.max(np.abs(y2))
        maxx = max([max1, max2])/1e16

        time2 = np.delete(time2, -1)
        y2 = np.delete(y2, -1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(15, 20)
        ax.set_ylim(-maxx, maxx)
        line1, = ax.plot(time1, y1/1e16)
        ax2 = ax.twinx()
        ax2.set_xlim(15, 20)
        ax2.set_ylim(-maxx, maxx)
        line2, = ax2.plot(time2, y2/1e16, 'r')
        plt.legend([line1, line2], ['Aug. 20', 'Aug. 21'])
        plt.xlabel('Time (Hours)')
        plt.ylabel('TECu')
        plt.title('TECu for PRN {0}'.format(satellite.prn))
        plt.show()



SixteenthHour = time(16, 0, 0)

prn_list = ['G02', 'G06', 'G12', 'G19', 'G24']
SatList = [Satellite(prn, observations1, observations2) for prn in prn_list]

plot_question5(SixteenthHour, *SatList)
