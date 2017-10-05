import numpy as np
from numpy import zeros, arange, floor, roll, conj, exp, pi, mean, unravel_index, std, log10
from types import SimpleNamespace
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from proj2_utils_script import *


class CoarseAcquire:

    def __init__(self, time, sampling_frequency):
        file_to_sim_data = './asen5090_gps-data_5MHz_0IF_complex.mat'
        data_length = int(sampling_frequency*time)
        self.simulated_data = scipy.io.loadmat(file_to_sim_data)['signal'][0]
        self.simulated_data -= mean(self.simulated_data)
        self.simulated_data = self.simulated_data[0:data_length]

    def circular_correlation(self, reference_signal, received_data):
        x, y = reference_signal, received_data
        assert (len(x) == len(y))
        N = len(x)
        correlation = zeros([N,], dtype=complex)
        for n in range(N):
            # np.vdot takes the complex conjugate of the first argument before performing the dot product
            correlation[n] = np.vdot(y, roll(x, -n))
        return correlation

    def get_doppler_bins(self, T, fd_min=-5000, fd_max=5000):
        '''Computes a set of Doppler bins necessary to test when performing
        coarse acquisition.  These values are determined by the integration
        time, as well as min and max Doppler values.
        `T` -- integration period (seconds)
        `fd_min` -- (default=-5000) minimum Doppler frequency (Hz)
        `fd_max` -- (default=5000) maximum Doppler frequency (Hz)
        '''
        delta_f = 1/T  # Frequency bin step size, should be in Hz so T must be given in s
        bins = arange(fd_min, fd_max, delta_f)
        return bins

    def coarse_acquire(self, prn, f_chip, T, fs=5e6, fi=0):
        '''Performs coarse acquisition algorithm on data `x` given BPSK
        signal defined by `code` and with carrier frequency that has
        been heterodyned to `fi` Hz.

        `self.simulated_data` -- the raw data samples
        `fs` -- sampling rate of `simulated_data` (Hz)
        `fi` -- intermediate frequency for the given signal
        `code` -- (0/1) BPSK code sequence
        `f_chip` -- BPSK code chipping rate
        `T` -- integration time (seconds)
        `doppler_bins` -- the Doppler frequency bins to search

        Returns: --
        `n0` -- sample phase (i.e. code phase in units of samples)
        `fd` -- the acquired Doppler frequency
        `snr` -- the signal-to-noise ratio of the 2D correlation
        `correlation` -- the 2D correlation magnitude array
        '''
        doppler_bins = self.get_doppler_bins(T)
        code_phase_bins = np.arange(0, fs*T)
        m = len(doppler_bins)
        n = int(fs * T)  # this is how many code phase choices I have
        correlation = zeros((m, n), dtype=complex)
        assert (len(self.simulated_data) >= n)
        time_vector = arange(n) / fs
        for doppler_index, doppler_frequency in enumerate(doppler_bins):
            CA = generate_code_samples(time_vector, prn, f_chip)
            carrier_ref = np.exp(1j * 2 * pi * doppler_frequency * time_vector, dtype=complex)
            CA_ref = CA * carrier_ref
            cross_corr = self.circular_correlation(CA_ref, self.simulated_data)
            correlation[doppler_index, :] = cross_corr
        doppler_max, code_max = np.unravel_index(np.absolute(correlation).argmax(),correlation.shape)
        # dop_max, cod_max = correlation.argmax(0)
        doppler_shift = doppler_bins[doppler_max]
        code_shift = code_phase_bins[code_max]
        acquisition = SimpleNamespace()
        acquisition.prn = prn
        acquisition.correlation = np.absolute(correlation)
        acquisition.doppler_shift = [doppler_max, doppler_shift]
        acquisition.code_shift = [code_max, code_shift]
        return acquisition

    def estimate_snr(self, acquisition, T, fs=5e6):
        doppler_samples = len(self.get_doppler_bins(T))
        phase_sample = T * fs
        correlation = acquisition.correlation
        doppler_shift_index = acquisition.doppler_shift[0]
        code_phase_shift_index = acquisition.code_shift[0]
        try:
            ps = (np.absolute(correlation[doppler_shift_index, code_phase_shift_index]) ** 2 +
                  np.absolute(correlation[doppler_shift_index, code_phase_shift_index -1]) ** 2 +
                  np.absolute(correlation[doppler_shift_index, code_phase_shift_index + 1]) ** 2) / 3
            pn = (np.sum(np.absolute(np.power(correlation, 2))) - ps)/ (phase_sample * doppler_samples - 3)
        except:
            ps = np.absolute(correlation[doppler_shift_index, code_phase_shift_index] ** 2)
            pn = (np.sum(np.absolute(np.power(correlation,2))) - ps)/(phase_sample * doppler_samples - 1)
        return 10 * log10(ps/pn)

    def surface_plot(self, acquisition, T):
        doppler_bins = self.get_doppler_bins(T)
        matrix = acquisition.correlation
        fig = plt.figure(figsize=(17,7))
        fig.subplots_adjust(wspace=.5)
        fig.suptitle("Coarse Aquistion for PRN {0}".format(prn))
        ax = fig.add_subplot(1,2,1, projection='3d')
        m, n = matrix.shape
        x, y = arange(n), arange(m)
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, matrix)
        ax.scatter(acquisition.code_shift[0], acquisition.doppler_shift[0],
                   matrix[acquisition.doppler_shift[0], acquisition.code_shift[0]], marker='o', color='r')
        ax.text(acquisition.code_shift[0]+1, acquisition.doppler_shift[0]+0.25,
                matrix[acquisition.doppler_shift[0], acquisition.code_shift[0]]+1,
                "Doppler Shift: " + str(acquisition.doppler_shift[1]))
        ax.text(acquisition.code_shift[0]+1, acquisition.doppler_shift[0]+0.25,
                matrix[acquisition.doppler_shift[0], acquisition.code_shift[0]]
                - .1*matrix[acquisition.doppler_shift[0], acquisition.code_shift[0]],
                "Code Shift: " + str(acquisition.code_shift[1]))
        ax.set_title('Coarse Acquisition Correlation Matrix Surface')
        ax.set_xlabel('Sample Phase')
        ax.set_ylabel('Doppler (Hz)')
        ax.set_yticks(range(0,len(doppler_bins)))
        ax.set_yticklabels([str(fd) for fd in doppler_bins])
        ax.set_zlabel('Correlation')

        ax = fig.add_subplot(1,2,2)
        n0 = acquisition.code_shift[1]
        N = matrix.shape[1]
        spread = 100
        n_start = int(max(0, min(n0 - spread, N - 2 * spread)))
        n_stop = int(min(n_start + 2 * spread, N))
        ax.imshow(matrix[:, n_start:n_stop], aspect='auto', interpolation='nearest',
                  extent=[n_start, n_stop, 10, -2])
        ax.set_yticks(range(0, len(doppler_bins)))
        ax.set_yticklabels([str(fd) for fd in doppler_bins])
        ax.set_title('Coarse Acquisition Correlation Matrix')
        ax.set_xlabel('Sample phase')
        ax.set_ylabel('Doppler (Hz)')
        plt.show()

if __name__ == '__main__':
    t = 0.001  # in seconds
    sampling_frequency = 5e6
    f_chip = 1.023e6  # MHz
    acquire = CoarseAcquire(t, sampling_frequency)
    prns = [4, 7, 10, 15]
    for prn in prns:
        acquisition = acquire.coarse_acquire(prn, f_chip, t)
        print("Estimated SNR for PRN {0}: {1} dB".format(prn, np.round(acquire.estimate_snr(acquisition, t),2)))
        acquire.surface_plot(acquisition, t)


