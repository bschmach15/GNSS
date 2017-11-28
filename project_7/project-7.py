
# coding: utf-8

# # ASEN 5090 Assignment 7

# In[ ]:

import numpy
from numpy import array, datetime64, exp, where, diff, nan, concatenate, ndarray
from types import SimpleNamespace
from datetime import datetime, timezone

from rinex_utilities import parse_rinex_obs_file
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.rcParams.update({'font.size': 20})


# In[ ]:

# CDDIS FTP path:  ftp://cddis.gsfc.nasa.gov/gnss/data/daily/2017/223/17o/
rinex_filename1 = #! FIND APPROPRRIATE RINEX FILE
rinex_filename2 = #! FIND APPROPRRIATE RINEX FILE
header1, observations1 = parse_rinex_obs_file(rinex_filename1)
header2, observations2 = parse_rinex_obs_file(rinex_filename2)


# In[ ]:

fL1 = 1.57542e9
fL2 = 1.2276e9
c = 299792458
kappa = #! COMPUTE KAPPA CONSTANT


# In[ ]:

#! YOUR CODE HERE
#! MAKE APPROPRIATE PLOTS

