import numpy as np
from numpy import zeros, ones, arange, floor, sum, roll
import pandas as pd
import matplotlib.pyplot as plt
import ast


class PRN:

    def __init__(self):
        """The main functions we were tasked with are at the beginning of the class. Any helper functions I wrote myself
        or any functions I used to check my work and verify things are at the end of the class. I will mark them with 
        doc strings like this with that they're used for"""
        # Load in the phase assignments as a dataframe so I know what I'm grabbing, helps me keep track of what
        # I'm doing where
        self.l1_code_phase_assignments = pd.read_csv("./l1_code_phase_assignments.csv",
                                                     converters={"CA_Phase_Select": ast.literal_eval})
        # set the g1 and g2 feedback taps as members for easy reference
        self.g1_feedback_taps = [3, 10]
        self.g2_feedback_taps = [2, 3, 6, 8, 9, 10]

    def generate_mls(self, N, feedback_taps, output_taps):
        """Generates maximum-length sequence (MLS) for the given linear feedback
        shift register (LFSR) length, feedback taps, and output taps.  The initial
        state of the LFSR is taken to be all ones.

        Parameters
        ----------------------------------------------------------------------------
        N : int
            length of LFSR
        feedback_taps : array or ndarray of shape (L,)
            the L taps to use for feedback to the shift register's first value
        output_taps : array or ndarray of shape (M,)
            the M taps to use for choosing the code output

        Returns
        ----------------------------------------------------------------------------
        output : ndarray of shape (2**N - 1,)
            the binary MLS values
        """
        shift_register = ones((N,))
        len_of_mls = 2**N-1
        values = zeros((len_of_mls,))
        # since python counts at 0, we need to account for output and feedback taps starting to count at 1
        # for example, if we want feedback tap position 3, that is position 2 in python.
        output_taps = [tap - 1 for tap in output_taps]
        feedback_taps = [tap - 1 for tap in feedback_taps]
        for i in range(len_of_mls):
            # calculate the feedback sum using modulo 2 addition
            feedback_sum = self.make_feedback(shift_register, feedback_taps)
            # add output bit to MLS
            values[i] = self.make_output_bit(shift_register, output_taps)
            # shift everything to the right by 1 space and fill the new space with the feedback sum
            shift_register = roll(shift_register,1)
            shift_register[0] = feedback_sum
        return values

    def generate_l1ca_codes(self, prn):
        """Generates GPS L1 C/A code for given PRN.

            Parameters
            ----------------------------------------------------------------------------
            prn : int 
                the signal PRN

            Returns
            ----------------------------------------------------------------------------
            output : ndarray of shape(1023,)
                the complete code sequence
            """
        output_taps = self.l1_code_phase_assignments.loc[prn, 'CA_Phase_Select']
        g1 = self.generate_mls(10, self.g1_feedback_taps, [10])
        g2 = self.generate_mls(10, self.g2_feedback_taps, output_taps)
        ca_code = []
        for index, bit in enumerate(g1):
            ca_code.append(int((bit + g2[index]) % 2))
        return ca_code

    def generate_code_samples(self, code, fs, T, fc, c0=0):
        """Generates samples of code sequence given sampling duration / rate and
        code sequence, code chipping rate, and initial code phase (optional).

        Parameters
        ----------------------------------------------------------------------------
        fs : float 
            sampling rate
        T : float
            duration of sampled signal (seconds)
        code : ndarray of shape (M,)
            code sequence to sample
        fc : float
            code chipping rate
        c0 : float
            (optional) defaults to zero -- the initial code phase in the sampled
            time-series

        Returns
        ----------------------------------------------------------------------------
        output : ndarray of shape (N,)
            the code samples
        """
        sample_interval = 1/fs #t_s
        chip_width = 1/fc # t_c in slides, in nano-seconds
        time_vector = np.arange(0, T, sample_interval)
        for index, time in enumerate(time_vector):
            # map_index is what chip out of the 1023 chips in the C/A code we will sample
            map_index = floor(time/chip_width) % 1023
            # replace the time_vector index with the chip sample from the C/A code
            time_vector[index] = code[int(map_index)]
        return time_vector

    def verify_l1ca_code_first_10_bits(self, prn):
        code = self.generate_l1ca_codes(prn)
        octal = self.l1_code_phase_assignments.loc[prn, 'CA_First_10_Chips']
        octal_str = '0o' + str(octal)
        first_10_bits = int(''.join([repr(int(i)) for i in code[:10]]), 2)
        return octal_str == oct(first_10_bits)

    def check_all_prns(self):
        results = {prn: self.verify_l1ca_code_first_10_bits(prn) for prn in range(1, 33)}
        passes = sorted([prn for prn, res in results.items() if res])
        failures = sorted([prn for prn, res in results.items() if not res])
        if len(passes) == 32:
            print('All PRNs passed')
            return True
        elif len(failures):
            print('PRNs {0} did NOT pass'.format(' '.join(failures)))
            if len(passes):
                print('PRNs {0} passed'.format(' '.join(passes)))
            return False

    def circular_correlation(self, x, y):
        N = len(x)
        z = zeros((N,))
        for n in range(N):
            z[n] = sum(x * roll(y, -n))
        return z

    def create_constant_magnitude_signal(self, samples):
        """Replace all 0's with -1s in the C/A code or in the code samples"""
        for index, value in enumerate(samples):
            if value == 0:
                samples[index] = -1
            else:
                continue
        return samples

    def plot_correlation(self, x,y, T, fs, prn1, prn2):
        """Plot the correlation values for the time sequence
        Auto-correlations for both code samples are plotted
        so are the cross-correlation values"""
        x = self.create_constant_magnitude_signal(x)
        y = self.create_constant_magnitude_signal(y)
        auto_1 = self.circular_correlation(x, x)
        auto_2 = self.circular_correlation(y, y)
        cross = self.circular_correlation(x, y)
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        t = arange(0, T, 1.0 / fs)
        ax1.plot(t, auto_1)
        ax1.set_title('Auto-correlation PRN {0}'.format(prn1))
        ax1.set_xlabel('Time (ms)')
        ax2.plot(t, auto_2)
        ax2.set_title('Auto-correlation PRN {0}'.format(prn2))
        ax2.set_xlabel('Time (ms)')
        ax3.plot(t, cross)
        ax3.set_title('Cross-correlation b/t PRN {0} and PRN {1}'.format(prn1, prn2))
        ax3.set_xlabel('Time (ms)')
        xlim = (0, T)
        ylim = ax1.get_ylim()
        for ax in [ax1, ax2, ax3]:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_xticklabels(['{0:1.1f}'.format(1e3 * t) for t in ax.get_xticks()])
        plt.show()

    def plot_correlation_ca_code(self, x, y, prn1, prn2):
        """A function to plot the correlation values for the C/A codes, or the gold codes.
         I wrote this primarily to make sure what I was producing was consistent with what was 
         shown in the lecture slides"""
        x = self.create_constant_magnitude_signal(x)
        y = self.create_constant_magnitude_signal(y)
        auto_1 = self.circular_correlation(x, x)
        auto_2 = self.circular_correlation(y, y)
        cross = self.circular_correlation(x, y)
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        t = arange(0, len(x))
        ax1.plot(t, auto_1)
        ax1.set_title('Auto-correlation PRN {0}'.format(prn1))
        ax1.set_xlabel('Sample Number')
        ax2.plot(t, auto_2)
        ax2.set_title('Auto-correlation PRN {0}'.format(prn2))
        ax2.set_xlabel('Sample Number')
        ax3.plot(t, cross)
        ax3.set_title('Cross-Correlation b/t PRN {0} and PRN {1}'.format(prn1, prn2))
        ax3.set_xlabel('Sample Number')
        xlim = (-10, len(x))
        ylim = ax1.get_ylim()
        for ax in [ax1, ax2]:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
        ax3.set_xlim(xlim)
        plt.show()

    def make_feedback(self,shift_register, feedback_taps):
        """This quickly calculates the feedback value to simplify the for-loop.
        For use in the generate_mls method"""
        feedback_list = []
        for tap in feedback_taps:
            feedback_list.append(shift_register[tap])
        return sum(feedback_list) % 2

    def make_output_bit(self, shift_register, output_taps):
        """This quickly calculates the output value to simplify the for-loop.
        For use in the generate_mls method"""
        output_list = []
        for tap in output_taps:
            output_list.append(shift_register[tap])
        return sum(output_list) % 2

    def check_correlations(self, x, y, prn1, prn2):
        """I wrote this one primarily to check to make sure that my values were consistent with what
        was shown in the lecture slides. I wanted to make sure that the correct values appeared 
        with the correct percentage (or close to)."""
        x = self.create_constant_magnitude_signal(x)
        y = self.create_constant_magnitude_signal(y)
        auto_1 = self.circular_correlation(x, x)
        auto_2 = self.circular_correlation(y, y)
        print("Max Value in Auto-Correlation for PRN {0}: {1}".format(prn1, max(auto_1)))
        print("Max Value in Auto-Correlation for PRN {0}: {1}".format(prn2, max(auto_2)))
        auto_1_63, auto_1_65, auto_1_minus_1 = np.sum(auto_1 == 63)/1023, np.sum(auto_1 == -65)/1023, (np.sum(auto_1 == -1)/1023)
        auto_2_63, auto_2_65, auto_2_minus_1 = np.sum(auto_2 == 63)/1023, np.sum(auto_2 == -65)/1023, np.sum(auto_2 == -1)/1023
        print("63 appears {0}% in Auto-Correlation for PRN {1}".format(auto_1_63 * 100, prn1))
        print("-65 appears {0}% in Auto-Correlation for PRN {1}".format(auto_1_65 * 100, prn1))
        print("-1 appears {0}% in Auto-Correlation for PRN {1}".format(auto_1_minus_1 * 100, prn1))
        print("63 appears {0}% in Auto-Correlation for PRN {1}".format(auto_2_63 * 100, prn2))
        print("-65 appears {0}% in Auto-Correlation for PRN {1}".format(auto_2_65 * 100, prn2))
        print("-1 appears {0}% in Auto-Correlation for PRN {1}".format(auto_2_minus_1 * 100, prn2))

    def main_function(self, prn1, prn2):
        """A function that performs everything that needs to be done with two different PRNs"""
        self.check_all_prns()
        fs = 5e6  # 5 MHz sampling rate
        T = 5e-3  # 5 milliseconds sampling duration
        fc = 1.023e6
        code_1 = self.generate_l1ca_codes(prn1)
        samples_1 = self.generate_code_samples(code_1, fs, T, fc)
        code_2 = self.generate_l1ca_codes(prn2)
        samples_2 = self.generate_code_samples(code_2, fs, T, fc)
        # To display correlation check, uncomment the next line
        # self.check_correlations(code_1, code_2, prn1, prn2)
        self.plot_correlation(samples_1, samples_2, T, fs, prn1, prn2)
        # To plot the CA code correlation plots, uncomment the next line
        # self.plot_correlation_ca_code(code_1, code_2, prn1, prn2)

if __name__ == '__main__':
    Psuedo = PRN() #initialize the class
    prn_1, prn_2 = 20, 21 # define the PRNs you would like to use
    Psuedo.main_function(prn_1, prn_2)
