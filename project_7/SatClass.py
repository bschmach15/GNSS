import pandas as pd
import numpy as np
import re

class Satellite:

    def __init__(self, prn, obs1, obs2):
        self.number = prn
        self.prn = re.search('\d+', self.number).group(0)
        self.fL1 = 1.57542e9
        self.fL2 = 1.2276e9
        self.c = 299792458
        aug_20_rinex, aug_21_rinex = self.two_day_rinex(prn, obs1, obs2)
        self.aug_20 = self.compute_TEC(aug_20_rinex)
        self.aug_21 = self.compute_TEC(aug_21_rinex)

    def two_day_rinex(self, prn, observations1, observations2):
        rinex_1 = observations1[prn]
        rinex_2 = observations2[prn]
        return rinex_1, rinex_2

    def make_dataframe(self, rinex):
        time = rinex.time
        L1 = rinex.signals['L1']
        L2 = rinex.signals['L2']
        dataframe = pd.DataFrame({"pr": L1.pr, "L1 SNR": L1.snr, "L2 SNR": L2.snr, "L1 Doppler": L1.doppler,
                                  "L2 Doppler": L2.doppler, 'L1 Carrier': L1.carrier, 'L2 Carrier': L2.carrier},
                                 index=time)
        return dataframe

    def compute_TEC(self, rinex):
        dataframe_rinex = self.make_dataframe(rinex)
        carrier_phase_L1, carrier_phase_L2 = dataframe_rinex['L1 Carrier'], dataframe_rinex['L2 Carrier']
        psuedorange_L1 = carrier_phase_L1 * (self.c / self.fL1)
        psuedorange_L2 = carrier_phase_L2 * (self.c / self.fL2)
        TEC = (1 / 40.3) * ((self.fL1 ** 2 * self.fL2 ** 2) / (self.fL1 ** 2 - self.fL2 ** 2)) * \
              (np.subtract(psuedorange_L1, psuedorange_L2))
        dataframe_rinex['TEC'] = TEC
        return dataframe_rinex

    def get_two_day_plot_items(self):
        time1 = self.aug_20.index.time
        time2 = self.aug_21.index.time
        for index, time in enumerate(time1):
            time1[index] = time.hour + time.minute / 60.0 + time.second / 3600.0
        for index, time in enumerate(time2):
            time2[index] = time.hour + time.minute / 60.0 + time.second / 3600.0
        y1 = self.aug_20['TEC'].values
        y2 = self.aug_21['TEC'].values
        return time1, y1, time2, y2

