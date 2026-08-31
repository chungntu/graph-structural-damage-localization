# -*- coding: utf-8 -*-
"""
main.py
An example script for extracting data from the h5 file (HBTA dataset).

"""

import h5py
import pandas as pd

from functions import extract_hdf5


# ------------------------------------ #
# HDF5 FILE
# ------------------------------------ #

# HDF5 file
h5_file = 'data_100Hz.h5'


# ------------------------------------ #
# EXTRACT DATA - Alternative 1
# ------------------------------------ #

# Extract data directly from the h5 file

# Import
data_h5 = h5py.File(h5_file, 'r')

# Variables
recording = 'MVS_P2_UDS_NM_Z_01'
sensor_group = 'acceleration'
sensor = 'AG01'
channel = 'y'

# DataFrame of a single channel
data_1_df = pd.DataFrame(data=data_h5[recording][sensor_group][sensor][channel],
                         columns=[sensor + '/' + channel])

# Information and view of data
print(data_1_df.info())
print()
print(data_1_df.head())


# ------------------------------------ #
# EXTRACT DATA - Alternative 2
# ------------------------------------ #

# Extract data using a convenient function

# Variables
recordings = ['MVS_P2_UDS_NM_Z_01', 'MVS_P2_UDS_SM_Z_01']

# Extract data
data = extract_hdf5(h5_file, recordings)

# DataFrame of all channels (of the first recording)
data_df = data[recordings[0]]['data_df']

# Information and view of data
print()
print(data_df.info())
print()
print(data_df.head())
