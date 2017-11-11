# ASEN 5090 Assignment 4
import numpy as np
from numpy import sin, cos, radians
from project_utilities import ecef2sky, geo2ecef
from types import SimpleNamespace
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

# You can ignore the `download_text` and `parse_yuma_almanacs` functions -- they're just here to make life easier.
def download_text(url):
    '''Downloads `url` and returns a string containing the response content.'''
    import urllib.request
    response = urllib.request.urlopen(url)
    data = response.read()      # a `bytes` object
    text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
    return text


def parse_yuma_almanacs(lines):
    '''Given the lines of a Yuma almanac file, produces a dictionary {sat_id: <namespace>},
    where `namespace` contains the almanac parameters.
    
    Input:
        `lines` -- list containing lines from a Yuma almanac file
    Output:
        `almanacs` -- dictionary {<sat_id>: <namespace>} where namespace contains:
            `Sat_ID` -- satellite ID or PRN
            `Health` -- satellite health indicator
            `Eccentricity` -- satellite orbit eccentricity
            `Time_of_Applicability` -- time of applicability of almanac ephemeris for satellite
            `Orbital_Inclination` -- satellite orbit inclination
            `Rate_of_RAAN` -- rate of change of RAAN in ECI frame
            `Sqrt_a` -- square root of orbit semi-major axis (m^(1/2))
            `RAAN_at_Week` -- RAAN at start of corresponding GPS week (rad.)
            `Arg_of_Perigee` -- orbit argument of perigee (rad.)
            `Mean_Anomaly` -- satellite mean anomaly (rad.)
            `Af0` -- coarse GPS clock correction bias term (s)
            `Af1` -- coarse GPS clock correction frequency term (s/s)
            `Week` -- GPS week number (mod 1024)
    '''
    index = 0
    L = len(lines)
    almanacs = {}
    while index < L:
        if lines[index].startswith('ID') and index < L - 13:
            alm = SimpleNamespace()
            alm.Sat_ID = int(lines[index][25:])
            alm.Health = int(lines[index + 1][25:])
            alm.Eccentricity = float(lines[index + 2][25:])
            alm.Time_of_Applicability = float(lines[index + 3][25:])
            alm.Orbital_Inclination = float(lines[index + 4][25:])
            alm.Rate_of_RAAN = float(lines[index + 5][25:])
            alm.Sqrt_a = float(lines[index + 6][25:])
            alm.RAAN_at_Week = float(lines[index + 7][25:])
            alm.Arg_of_Perigee = float(lines[index + 8][25:])
            alm.Mean_Anomaly = float(lines[index + 9][25:])
            alm.Af0 = float(lines[index + 10][25:])
            alm.Af1 = float(lines[index + 11][25:])
            alm.Week = int(lines[index + 12][25:])
            almanacs[alm.Sat_ID] = alm
            index += 13
        else:
            index += 1
    return almanacs


def solve_kepler(M, e, E0=0, tol=1e-12, timeout=200):
    '''Given mean anomaly `M` and orbital eccentricity `e`, uses Newton's method
    to iteratively solve Kepler's equations for the eccentric anomaly.
    Iteration stops when the magnitude between successive values of eccentric
    anomaly `E` is less than `tol`, or when `timeout` iterations have occurred in
    which case an exception is thrown.
    
    Input:
        `M` -- mean anomaly as scalar or array of shape (N,)
        `e` -- eccentricity (scalar)
        `E0` -- initial guess for `E` as scalar or array of shape (N,)
        `tol` -- desired tolerance for `abs(E[k+1]-E[k]) < tol`
        `timeout` -- number of iterations to attempt before throwing timeout error
    Output:
        `E` -- the eccentric anomaly as scalar or array of shape (N,)
    '''
    eccentric_anomaly = np.copy(M)
    for n, m in enumerate(M):
        E = E0
        difference = E - e * sin(E) - m
        while abs(difference) > tol:
            E = E - difference/(1 - e * cos(E))
            difference = E - e * sin(E) - m
            timeout -= 1
            if timeout <= 0:
                break
        eccentric_anomaly[n] = E
    return eccentric_anomaly


def compute_ecef_position_from_gps_almanac(alm, t, rollover=1, mu_E=3.986005e14, Omega_E_dot=7.2921151467e-5):
    '''Given Yuma almanac for a particular SV and a scalar or an array of times,
    extracts the necessary Keplerian parameters and computes satellite position
    in ECEF coordinates.  Note that, because ECEF is a rotating coordinate frame,
    RAAN (denoted `Omega`) will depend on time `t`, as will the true anomaly `nu`.
    
    Input:
        `alm` -- almanac namespace output from `parse_yuma_almanacs`. This
            almanac corresponds to one satellite.
        `t` -- time array of shape (N,) of GPST seconds. Seconds since 00:00:00 Jan 6 1980
        `rollover` -- (optional) if almanac specifies week number modulo-1024,
            this rollover indicates how many multiples of 1024 to add to get the
            actual week number.
        `mu_E` -- (optional) Earth gravitational parameter (m^3/s^2)
        `Omega_E_dot` -- (optional) WGS Earth rotation rate
    Output:
        array of shape (N, 3) of satellite ECEF coordinates
    '''

    #! first extract the Keplerian elements `a`, `e`, `i`, `omega` from almanac
    a, e, i, omega = alm.Sqrt_a ** 2, alm.Eccentricity, alm.Orbital_Inclination, alm.RAAN_at_Week

    #! next compute the time of week `TOW` corresponding to the GPST input `t`
    #! and compute `Dt` -- the difference between `TOW` and the almanac
    #! ephemeris time of applicability

    seconds_in_week = 60.0*60.0*24.0*7.0
    week_number = np.floor(t / seconds_in_week)
    time_of_week = t - (week_number * seconds_in_week)
    toa = alm.Time_of_Applicability
    Dt = time_of_week - toa

    #! next, compute `Omega` (the RAAN) in the ECEF coordinate frame

    raan_rate = alm.Rate_of_RAAN
    RAAN = omega - Omega_E_dot * time_of_week + raan_rate * Dt

    #! compute mean motion, mean anomaly, and eccentric anomaly

    mean_motion = np.sqrt(mu_E/(a ** 3))
    mean_anomaly = alm.Mean_Anomaly + mean_motion * Dt
    eccentric_anomalies = solve_kepler(mean_anomaly, e)

    #! compute true anomaly, orbital radius

    true_anomalies = np.arctan(np.divide(np.sqrt(1 - e ** 2) * sin(eccentric_anomalies),cos(eccentric_anomalies) - e))
    orbital_radii = a * (1 - e * cos(eccentric_anomalies))

    #! compute preliminary (x, y) orbit coordinates (follow steps from lecture 14 slide 19)
    #! using so-called "argument of latitude" (`omega + nu`), which will account for the 
    #! rotation by `omega` that we would normally do to convert from perifocal to ECEF frame

    x_p = orbital_radii * cos(Omega_E_dot + true_anomalies)
    y_p = orbital_radii * sin(Omega_E_dot + true_anomalies)

    #! apply rotations `R3(Omega)R1(i)` to the (x, y) coordinates derived above
    x_t = x_p * cos(RAAN) - y_p * sin(RAAN) * cos(i)
    y_t = x_p * sin(RAAN) + y_p * cos(RAAN) * cos(i)
    z_t = y_p * sin(i)

    return np.hstack((x_t.reshape(-1,1), y_t.reshape(-1,1), z_t.reshape(-1,1)))


# On October 31 of last year (2016), your friend -- who was in Boulder at the time -- claims that they
# saw a GPS satellite due North at an elevation of $30^\circ$ above the horizon.  This was sometime
# around 10:30 PM MST.  Is your friend correct?  Or did they just see...  a   *s P o O k y*  ghost
# satellite?  Generate a skyplot for all GPS satellites and for times 9:00 - 11:30 PM MST at 5-minute
# intervals and use it to support your claim.  9:00 PM MST corresponds to 1162004417.0 seconds GPST.


def make_sky_plot(almanac, obs_ecef, times_gpst):
    # make sky plot for computed satellite positions
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    angles_list = [i for i in range(0,360,30)]
    elevation_list = [i for i in range(0,105,15)]
    ax.set_thetagrids(angles_list)
    ax.set_yticks(elevation_list)
    ax.set_yticklabels([str(i) + '\u00b0' for i in elevation_list[::-1]])
    ax.set_xticklabels(['North\n0\u00b0', '30\u00b0', '60\u00b0', 'East\n90\u00b0', '120\u00b0', '150\u00b0',
                        'South\n180\u00b0', '210\u00b0', '240\u00b0', 'West\n270\u00b0', '300\u00b0', '330\u00b0'])
    ax.set_ylim(0,90)

    for num, alm in almanac.items():
        sat = ecef2sky(obs_ecef, compute_ecef_position_from_gps_almanac(alm, times_gpst))
        az = radians(sat[:, 0])
        el = 90 - sat[:, 1]
        ax.plot(az, el, 'o', markerfacecolor='w')
        ax.annotate(str(alm.Sat_ID), xy=(az[-1], el[-1]))

    plt.show()


# the correct almanac file can be downloaded here:
almanac_url = 'http://www.celestrak.com/GPS/almanac/Yuma/2016/almanac.yuma.week0897.061440.txt'
# if you aren't running Python 3, this download won't work and you'll have to handle it some other way
almanac_text = download_text(almanac_url)
almanac_lines = almanac_text.split('\n')
almanacs = parse_yuma_almanacs(almanac_lines)
times = np.arange(1162004417.0, 1162013417, 300)

boulder_ecef = np.array([-1288648., -4720213., 4080224.])

make_sky_plot(almanacs, boulder_ecef, times)


