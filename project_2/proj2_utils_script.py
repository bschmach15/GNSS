# coding: utf-8

L1_CODE_PHASE_ASSIGNMENTS = {
    # prn : svn, prn, ca_phase_select, x2_phase_select, ca_code_delay, p_code_delay, ca_first_10_chips, p_first_12_chips
    1: (1, 1, (2, 6), 1, 5, 1, 1440, 4444),
    2: (2, 2, (3, 7), 2, 6, 2, 1620, 4000),
    3: (3, 3, (4, 8), 3, 7, 3, 1710, 4333),
    4: (4, 4, (5, 9), 4, 8, 4, 1744, 4377),
    5: (5, 5, (1, 9), 5, 17, 5, 1133, 4355),
    6: (6, 6, (2, 10), 6, 18, 6, 1455, 4344),
    7: (7, 7, (1, 8), 7, 139, 7, 1131, 4340),
    8: (8, 8, (2, 9), 8, 140, 8, 1454, 4342),
    9: (9, 9, (3, 10), 9, 141, 9, 1626, 4343),
    10: (10, 10, (2, 3), 10, 251, 10, 1504, 4343),
    11: (11, 11, (3, 4), 11, 252, 11, 1642, 4343),
    12: (12, 12, (5, 6), 12, 254, 12, 1750, 4343),
    13: (13, 13, (6, 7), 13, 255, 13, 1764, 4343),
    14: (14, 14, (7, 8), 14, 256, 14, 1772, 4343),
    15: (15, 15, (8, 9), 15, 257, 15, 1775, 4343),
    16: (16, 16, (9, 10), 16, 258, 16, 1776, 4343),
    17: (17, 17, (1, 4), 17, 469, 17, 1156, 4343),
    18: (18, 18, (2, 5), 18, 470, 18, 1467, 4343),
    19: (19, 19, (3, 6), 19, 471, 19, 1633, 4343),
    20: (20, 20, (4, 7), 20, 472, 20, 1715, 4343),
    21: (21, 21, (5, 8), 21, 473, 21, 1746, 4343),
    22: (22, 22, (6, 9), 22, 474, 22, 1763, 4343),
    23: (23, 23, (1, 3), 23, 509, 23, 1063, 4343),
    24: (24, 24, (4, 6), 24, 512, 24, 1706, 4343),
    25: (25, 25, (5, 7), 25, 513, 25, 1743, 4343),
    26: (26, 26, (6, 8), 26, 514, 26, 1761, 4343),
    27: (27, 27, (7, 9), 27, 515, 27, 1770, 4343),
    28: (28, 28, (8, 10), 28, 516, 28, 1774, 4343),
    29: (29, 29, (1, 6), 29, 859, 29, 1127, 4343),
    30: (30, 30, (2, 7), 30, 860, 30, 1453, 4343),
    31: (31, 31, (3, 8), 31, 861, 31, 1625, 4343),
    32: (32, 32, (4, 9), 32, 862, 32, 1712, 4343),
    33: (65, 33, (5, 10), 33, 863, 33, 1745, 4343),
    34: (66, 34, (4, 10), 34, 950, 34, 1713, 4343),
    35: (67, 35, (1, 7), 35, 947, 35, 1134, 4343),
    36: (68, 36, (2, 8), 36, 948, 36, 1456, 4343),
    37: (69, 37, (4, 10), 37, 950, 37, 1713, 4343),
}

from numpy import zeros, ones, arange, floor, sum, roll


def generate_mls(N, feedback_taps, output_taps):
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
        the M taps to use for choosing the sequence output

    Returns
    ----------------------------------------------------------------------------
    output : ndarray of shape (2**N - 1,)
        the binary MLS values
    """
    shift_register = ones((N,))
    values = zeros((2 ** N - 1,))
    for i in range(2 ** N - 1):
        values[i] = sum(shift_register[output_taps]) % 2
        first = sum(shift_register[feedback_taps]) % 2
        shift_register[1:] = shift_register[:-1]
        shift_register[0] = first
    return values


def generate_l1ca_code(prn):
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
    ps = L1_CODE_PHASE_ASSIGNMENTS[prn][2]
    g1 = generate_mls(10, [2, 9], [9])
    g2 = generate_mls(10, [1, 2, 5, 7, 8, 9], [ps[0] - 1, ps[1] - 1])
    return (g1 + g2) % 2


def generate_code_samples(t, prn, f_chip, c0=0):
    """Generates samples of code sequence given sampling duration / rate and
    code sequence, code chipping rate, and initial code phase (optional).

    Parameters
    ----------------------------------------------------------------------------
    t : ndarray of shape (N,)
        sampling times
    code : ndarray of shape (M,)
        code sequence to sample
    f_chip : float
        code chipping rate
    c0 : float
        (optional) defaults to zero -- the code phase at (t=0)

    Returns
    ----------------------------------------------------------------------------
    output : ndarray of shape (N,)
        the code samples
    """
    code = generate_l1ca_code(prn)
    chip_indices = floor(c0 + t * f_chip) % len(code)
    return code[chip_indices.astype(int)]


