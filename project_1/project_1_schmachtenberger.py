import numpy as np
from numpy import zeros, ones, arange, floor, sum, roll
import pandas as pd
import matplotlib.pyplot as plt
import ast

class PRN:

    def __init__(self):
        # Load in the phase assignments as a dataframe so I know what I'm grabbing, helps me keep track of what
        # I'm doing where
        self.l1_code_phase_assignments = pd.read_csv("./l1_code_phase_assignments.csv",
                                                     converters={"CA_Phase_Select":
                                                                     ast.literal_eval})
        # set the g1 and g2 feedback taps as members for easy reference
        self.g1_feedback_taps = [3, 10]
        self.g2_feedback_taps = [2, 3, 6, 8, 9, 10]

    def make_feedback(self,shift_register, feedback_taps):
        """This quickly calculates the feedback value to simplify the for-loop"""
        feedback_list = []
        for tap in feedback_taps:
            feedback_list.append(shift_register[tap])
        return sum(feedback_list) % 2

    def make_output_bit(self, shift_register, output_taps):
        """This quickly calculates the output value to simplify the for-loop"""
        output_list = []
        for tap in output_taps:
            output_list.append(shift_register[tap])
        return sum(output_list) % 2

    def generate_mls(self, N, feedback_taps, output_taps):
        shift_register = ones((N,))
        len_of_mls = 2**N-1
        values = zeros((len_of_mls,))
        # since python counts at 0, we need to account for output taps starting to count at 1
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
        output_taps = self.l1_code_phase_assignments.loc[prn, 'CA_Phase_Select']
        g1 = self.generate_mls(10, self.g1_feedback_taps, [10])
        g2 = self.generate_mls(10, self.g2_feedback_taps, output_taps)
        ca_code = []
        for index, bit in enumerate(g1):
            ca_code.append(int((bit + g2[index]) % 2))
        return ca_code

    def generate_code_samples(self, fs, T, code, fc, c0=0):
        pass

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
        elif len(failures):
            print('PRNs {0} did NOT pass'.format(' '.join(failures)))
            if len(passes):
                print('PRNs {0} passed'.format(' '.join(passes)))

    def circular_correlation(self, x, y):
        N = len(x)
        z = zeros((N,))
        for n in range(N):
            z[n] = sum(x * roll(y, -n))
        return z

    def replace_minus_1_with_zero(self, samples):
        for index, value in enumerate(samples):
            if value == 0:
                samples[index] = -1
            else:
                continue
        return samples

if __name__ == '__main__':
    Psuedo = PRN()
    # print(prn.generate_mls(10, (3,10), [10]))
    Psuedo.check_all_prns()

