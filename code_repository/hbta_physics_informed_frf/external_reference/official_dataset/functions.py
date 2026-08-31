# -*- coding: utf-8 -*-
"""
functions.py
Assembly of functions.

"""
import h5py
import pandas as pd


def extract_hdf5(h5_file, recordings):
    """Extraction of HDF5 data from the HBTA dataset.

    This function imports a HDF5 file. It returns a dictionary containing all
    sensor data from the specified recordings provided in the input.

    Parameters
    ----------
    h5_file: str
        A str containing the name of the HDF5 file (and location).
    recordings : list
        A list of recordings provided in strings.

    Returns
    -------
    data : dict
        A dictionary containing all sensor data for each recording.
    """

    # ------------------------------------ #
    # PARAMETERS
    # ------------------------------------ #

    # Import
    data_h5 = h5py.File(h5_file, 'r')

    # ACCELERATION DATA
    # List of sensors
    accelerometer_global_list = ['AG01', 'AG02', 'AG03', 'AG04', 'AG05',
                                 'AG06', 'AG07', 'AG08', 'AG09', 'AG10',
                                 'AG11', 'AG12', 'AG13', 'AG14', 'AG15',
                                 'AG16', 'AG17', 'AG18']

    accelerometer_local_list = ['AL01', 'AL02', 'AL03', 'AL04',
                                'AL05', 'AL06', 'AL07', 'AL08',
                                'AL09', 'AL10', 'AL11', 'AL12',
                                'AL13', 'AL14', 'AL15', 'AL16',
                                'AL17', 'AL18', 'AL19', 'AL20',
                                'AL21', 'AL22', 'AL23', 'AL24',
                                'AL25', 'AL26', 'AL27', 'AL28',
                                'AL29', 'AL30', 'AL31', 'AL32',
                                'AL33', 'AL34', 'AL35', 'AL36',
                                'AL37', 'AL38', 'AL39', 'AL40']

    accelerometer_MVS = ['AS']

    # List of all sensors
    accelerometer_list = (accelerometer_global_list +
                          accelerometer_local_list + accelerometer_MVS)

    data = {}

    # ------------------------------------ #
    # DATA
    # ------------------------------------ #

    for recording in recordings:

        # Acceleration data
        data_1_df = pd.DataFrame()

        # Add data
        for accelerometer in accelerometer_list:

            # Global sensors
            if accelerometer[:2] == 'AG':

                # Accelerometer AG09 has only data in z-direction
                if accelerometer == 'AG09':

                    # Data in z-direction
                    data_1_df[accelerometer + '/z'] = pd.DataFrame(data_h5[recording]
                                                                   ['acceleration']
                                                                   [accelerometer]
                                                                   ['z'])

                else:

                    # Data in y-direction
                    data_1_df[accelerometer + '/y'] = pd.DataFrame(data_h5[recording]
                                                                   ['acceleration']
                                                                   [accelerometer]
                                                                   ['y'])

                    # Data in z-direction
                    data_1_df[accelerometer + '/z'] = pd.DataFrame(data_h5[recording]
                                                                   ['acceleration']
                                                                   [accelerometer]
                                                                   ['z'])

            # Local sensors
            if accelerometer[:2] == 'AL':

                # Data in z-direction
                data_1_df[accelerometer + '/z'] = pd.DataFrame(data_h5[recording]
                                                               ['acceleration']
                                                               [accelerometer]
                                                               ['z'])

            # MVS sensor
            if accelerometer == 'AS':

                # Data in y-direction
                if recording.split('_')[4] == 'Y':

                    data_1_df[accelerometer + '/y'] = pd.DataFrame(data_h5[recording]
                                                                   ['acceleration']
                                                                   [accelerometer]
                                                                   ['y'])

                # Data in z-direction
                if recording.split('_')[4] == 'Z':

                    data_1_df[accelerometer + '/z'] = pd.DataFrame(data_h5[recording]
                                                                   ['acceleration']
                                                                   [accelerometer]
                                                                   ['z'])

        # STRAIN DATA
        # List of sensors
        strain_gage_list = ['SB01', 'SB02', 'SB03', 'SB04',
                            'SB05', 'SB06', 'SB07', 'SB08',
                            'SC01', 'SC02', 'SC03', 'SC04',
                            'SC05', 'SC06', 'SC07']

        # Strain data
        data_2_df = pd.DataFrame()

        # Add data
        for strain_gage in strain_gage_list:

            # Strain gages on the lower chord beams
            if strain_gage[:2] == 'SB':

                # Data in x-direction
                data_2_df[strain_gage + '/x'] = pd.DataFrame(data_h5[recording]
                                                             ['strain']
                                                             [strain_gage]
                                                             ['x'])

            # Strain gages on the transverse (cross) girders
            elif strain_gage[:2] == 'SC':

                # Data in y-direction
                data_2_df[strain_gage + '/y'] = pd.DataFrame(data_h5[recording]
                                                             ['strain']
                                                             [strain_gage]
                                                             ['y'])

        # ALL DATA
        data_df = pd.concat([data_1_df, data_2_df], axis=1)

        # Update dictionary
        data.update({recording: {'data_acceleration_df': data_1_df,
                                 'data_strain_df': data_2_df,
                                 'data_df': data_df}})

    return data
