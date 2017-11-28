# coding: utf-8
# # ASEN 5090 Assignment 6
import numpy
from numpy import array, datetime64, exp, sin, tan
from datetime import datetime
from coordinate_utilities import ecef2sky, geo2ecef
from orbit_utilities import parse_yuma_almanacs, compute_ecef_position_from_gps_almanac
from rinex_utilities import parse_rinex_obs_file
import matplotlib.pyplot as plt


# Yuma URL:  http://www.celestrak.com/GPS/almanac/Yuma/2017/almanac.yuma.week0906.061440.txt
almanac_filename = 'almanac.yuma.week0906.061440.txt'
with open(almanac_filename, 'r') as f:
    almanacs = parse_yuma_almanacs(f.readlines())

# CDDIS FTP path:  ftp://cddis.gsfc.nasa.gov/gnss/data/daily/2017/001/17o/
header, rinex = parse_rinex_obs_file('./nist0010.17o')

prn = 24
sat_id = 'G{0:02}'.format(prn)

almanac = almanacs[prn]
sat = rinex['G24']

gpst_epoch = datetime(1980,1,6,0,0,0,0)  # Jan 6th, 00:00:00 (midnight) with microseconds
start_of_day_utc = datetime(2017, 1, 1)

# Compute the GPST time for the start of the day by taking the difference between the GPST epoch
# and the start of day in terms of seconds.
gpst_start_of_day = (datetime64(start_of_day_utc) - datetime64(gpst_epoch)).astype('timedelta64[s]')

# Note: datetime64 timedelta is in microseconds
t = (sat.time - datetime64(start_of_day_utc)).astype('timedelta64[s]')  # Turn t into seconds

# Add start of GPST day to seconds into day to find GPST at each time from RINEX and then
# convert to floats
t = (t + gpst_start_of_day).astype(float)
sat_ecef = compute_ecef_position_from_gps_almanac(almanac, t)



def arden_buck(temp, pressure, relative_humidity):
    vapor_pressure = (1.0007 + 3.46e-6 * pressure) * (6.1121 * exp((17.502 * temp)/(240.97 + temp)))
    return vapor_pressure * relative_humidity


def troposphere_delay(temp, pressure, partial_pressure):
    """I used the Hopfield Model"""
    temp = temp + 273  # convert celsius to kelvin
    # Pressure must be total pressure
    hw, hd = 12, 43  # km
    Tzd, Tzw = 77.6e-6 * (pressure/temp) * (hd/5.0), .0373 * (partial_pressure/(temp ** 2)) * (hw/5.0)
    return Tzd, Tzw

def mapping(Tzd, Tzw, sky):
    el = sky[:,1]
    tan_el = tan(el)
    sin_el = sin(el)
    md = 1/(sin_el + (0.00143/(tan_el + 0.0445)))
    mw = 1/(sin_el + (0.00035/(tan_el + 0.017)))
    return Tzd * md + Tzw * mw

boulder_geo = array((40.0150, -105.2705, 1700))
sat_sky = ecef2sky(geo2ecef(boulder_geo), sat_ecef)

# Historical data gathered from
# https://www.wunderground.com/history/airport/KBDU/2017/1/1/DailyHistory.html?req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=

temp = 46  # degrees F
temp_c = 7.778  # degrees C
pressure = 1002.0 # millibars corrected to sea level
humidity = 0.56

partial_pressure = arden_buck(temp_c, pressure, humidity)
Tzd, Tzw = troposphere_delay(temp_c, pressure, partial_pressure)

Troposphere_Delay = mapping(Tzd, Tzw, sat_sky)
psuedorange = sat.signals['L1'].pr


def plot_stuff():

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(12,5))
    fig.subplots_adjust(wspace=.5)
    fig.suptitle('RINEX Data for PRN 24')
    x = t

    # Delay Plot
    ax = fig.add_subplot(1, 3, 1)
    y = Troposphere_Delay
    ax.set_title('Troposphere Delay')
    ax.set_xlabel('Time (GPST)')
    ax.set_ylabel('Delay (m)')
    ax.plot(x, y)
    # ax.set_xlim(x[0], x[-1])

    # Elevation Plot
    ax = fig.add_subplot(1, 3, 2)
    y = sat_sky[:,1]
    ax.set_title('Satellite Elevation')
    ax.set_xlabel('Time (GPST)')
    ax.set_ylabel('Elevation Angle')
    ax.plot(x, y)

    # Ratio Plot
    ax = fig.add_subplot(1, 3, 3)
    y = numpy.divide(Troposphere_Delay, psuedorange)
    ax.set_title('Delay Ratio')
    ax.set_xlabel('Time (GPST)')
    ax.set_ylabel('Ratio')
    ax.plot(x, y)

    plt.show()

plot_stuff()
