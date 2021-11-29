import pandas as pd
import numpy as np
import h5py
import os
import netCDF4 as nc


def load_CML_data(date_selected, provider):
    # TODO: add functionality of selecting multiple days at once (will take a lot of time!)
    # TODO: add compatibility with numpy arrays (PyTorch package needs numpy arrays for tensors)
    # This function loads the specific CML data from a specified day. It takes two variables
    # - date_selected (string): the date to be extracted from the dataset, format YYYYMMDD
    # - provider (string): the name of the data provider (NOKIA or NEC possible)

    # Create the right folder URL for the right dataset and specify the headers
    folder_URL_NEC = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NEC_data/NEC_data/'
    folder_URL_NOKIA = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NOKIA_data/NOKIA_data/'
    header_names = ['SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON', 'FREQ', 'DATE', 'RXMIN', 'RXMAX',
                    'DIST', 'P4_start.v', 'P4_start.u', 'P4_end.v', 'P4_end.u', 'PROV']

    # Select the right data location based on the provider and the date_selected
    if provider == 'NOKIA':
        spec_URL = folder_URL_NOKIA + 'nokia_fh_' + date_selected + '.txt'
    elif provider == 'NEC':
        spec_URL = folder_URL_NEC + 'NEC_' + date_selected + '.txt'

    # Load the dataset
    data = pd.read_csv(spec_URL, header=None, delim_whitespace=True, names=header_names)

    # Add a unique identifier for each link in the dataset
    data.insert(0, "ID", value=np.nan)
    data['ID'] = data.groupby(['SITE_LAT', 'SITE_LON', 'FAR_LAT',
                               'FAR_LON']).ngroup() + 1  # Start ID at 1, by adding +1 to the groupby function
    # Add average Longitude and Latitude for the middle of the link
    data['AVG_LON'] = (data['FAR_LON'] + data['SITE_LON']) / 2
    data['AVG_LAT'] = (data['FAR_LAT'] + data['SITE_LAT']) / 2

    return (data)


def split_data(dataset, train_perc, test_perc, val_perc=0):
    # TODO: make this function compatible with numpy arrays, add multiple strategies of splitting the data (spatial or temporal)
    # This function splits the supplied data in three different sets (or two if val_perc is not supplied) and returns those
    # It takes four arguments
    # - dataset (dataframe): dataset to be split
    # - train_perc (int): percentage of the dataset that will be used for training
    # - test_perc (int): percentage of the dataset that will be used for testing
    # - val_perc (int): percentage of the dataset that will be used for validating

    # Check if the percentages match and make sure that none of the percentages are negative
    if train_perc + test_perc + val_perc != 100:
        raise ValueError('The percentages do not add up to 100%. Check your numbers again')
    elif (train_perc or test_perc or val_perc) < 0:
        raise ValueError('Insert only positive integers as percentages')
    else:
        # Split the data here
        train_idx = round(len(dataset) * (train_perc / 100))
        test_idx = round(len(dataset) * (test_perc / 100)) + train_idx
        train_data = dataset.iloc[0:train_idx, :]
        test_data = dataset.iloc[train_idx:test_idx, :]
        val_data = dataset.iloc[test_idx:, :]

    # Sort the datasets by links ID
    train_data = train_data.sort_values(['ID','DATE'])
    test_data  = test_data.sort_values(['ID', 'DATE'])
    val_data   = val_data.sort_values(['ID','DATE'])
    return (train_data, test_data, val_data)


def load_radar_data_h5(date_selected, time_selected):
    # This function loads in the gridded radar data for a specific day
    # - date_selected (int); selected date, format YYYMMDD
    # - time_selected (int); selected time, format HHMM

    # Standard URL
    base_URL = 'C:/Users/ludod/Documents/MSc Thesis/2011/2011/'
    # Select the month from the selected date to go to the right folder
    month = str(date_selected)[4:6]

    # Add up the three 5-minute intervals that make up one 15 minute CML interval
    total_radar = np.zeros(shape=[765, 700], dtype='uint16')

    for i in [1, 2, 3]:
        if str(time_selected)[2:4] == '00':
            time_selected_prior = time_selected - 40 - i * 5
        else:
            time_selected_prior = time_selected - i * 5

        data_URL = base_URL + month + '/' + 'RAD_NL25_RAC_5min_' + str(date_selected) + str(
            time_selected_prior) + '_cor.h5'
        f = h5py.File(data_URL, mode='r')
        dataset = f.get('image1').get('image_data')
        total_radar = np.add(total_radar, dataset)

    # Load the dataframe with the coordinate data for the gridded radar rainfall set
    cwd = os.getcwd()
    coord_grid = pd.read_csv(os.path.join(cwd, "InterpolationGrid.dat"))

    return (total_radar, f, coord_grid)


def load_radar_data_netcdf(date_selected, time_selected):
    # This function loads the radar data from a specific time in a netCDF format.
    # - date_selected (int): selected date
    # - time_selected (int): selected time

    base_URL = 'C:/Users/ludod/Documents/MSc Thesis/RADNL_CLIM_EM_MFBSNL25_05m_20101231T235500_20111231T235500_netCDF4_0002/2011/'
    # Select the month from the selected date to go to the right folder
    month = str(date_selected)[4:6]
    # RAD_NL25_RAC_MFBS_EM_5min_201106010000
    # spec_URL = base_URL + month + '/RAD_NL25_RAC_MFBS_EM_5min_'+f'{date_selected:04d}'+f'{time_selected:04d}'+'.nc'

    # The data needs to be aggregated to 15 minutes.
    # TODO: implement functionality to load data from the previous day
    for i in [1, 2, 3]:
        if time_selected == 0000:  # TODO: fix this part, difficult to deal with the string and integer formats
            date_selected = date_selected - 1
            time_selected = time_selected + 2400
        if str(time_selected).zfill(4)[2:4] == '00':
            time_selected_prior = time_selected - 40 - i * 5
        else:
            time_selected_prior = time_selected - i * 5

        spec_URL = base_URL + month + '/RAD_NL25_RAC_MFBS_EM_5min_' + str(date_selected) + str(
            time_selected_prior).zfill(4) + '.nc'
        temp_dataset = nc.Dataset(spec_URL)
        temp_precp = temp_dataset['image1_image_data'][0, :, :]

        if i == 1:
            total_radar = temp_precp
        else:
            total_radar = total_radar + temp_precp

    return (total_radar, temp_dataset)


def ReadRainLocation(CML_data):
    # This function takes the CML data with time and AVG_LON and AVG_LAT of the link and returns a dataset with an extra column
    # that represents the true value for the precipitation.
    # TODO: Don't use the middle of the link but a weighted average of all pixels it passes through.
    # TODO: Store the combined dataset on a disk somewhere to avoid loading it over and over
    # Load the grid with the coordinates for the radar.
    cwd = os.getcwd()
    coord_grid = pd.read_csv(os.path.join(cwd, "radarcoordinaten_NL25_1km2_WGS84_full.dat"))
    print('Coordinates loaded')
    # The coordinates for each row of the dataset change slightly from row to row as the Earth curves. The spread in
    # these coordinates however is smaller than the difference from column to column, so an average is taken for each grid cell.
    # This ensures that the right column and row are selected for the right CML

    lon_grid = coord_grid['lon'].values.reshape(765, 700).mean(axis=0)
    lat_grid = coord_grid['lat'].values.reshape(765, 700).mean(axis=1)

    # Create a 'unique' list of times in the dataset
    times_inCML = CML_data['DATE'].unique()
    times_inCML.sort()
    match_set = CML_data[['ID', 'AVG_LON', 'AVG_LAT']].drop_duplicates(
        ignore_index=True)  # Dataset that stores the indices for each ID of the corresponding
    # grid cell in the radar image.
    match_set['X_radar'] = 0
    match_set['Y_radar'] = 0

    print('Matching set created. Size:', match_set.shape)
    # Calculate the distance to all rows and columns and take the smallest of these distances.

    for i in range(0, len(match_set)):
        Dist_lon = np.sqrt(np.power((lon_grid - match_set['AVG_LON'][i]), 2))
        Dist_lat = np.sqrt(np.power((lat_grid - match_set['AVG_LAT'][i]), 2))
        match_set.at[i, 'X_radar'] = Dist_lon.argmin()
        match_set.at[i, 'Y_radar'] = Dist_lat.argmin()
        # print(i,' added correctly')

    current_index = 0  # Keep track of where you are in the list
    CML_data['TARG_PRCP'] = 0.0000  # Create an empty column to be filled later
    CML_data = CML_data.sort_values(by='DATE')

    # Loop over all timesteps in the file
    for i in times_inCML:
        radar_set, _ = load_radar_data_netcdf(int(str(i)[0:8]), int(str(i)[8:12]))
        print('CML Current index and i:',current_index, CML_data['DATE'].iloc[current_index], i)

        # The while loop ensures that the for loop continues to the next time step as all links have been added
        while CML_data['DATE'].iloc[current_index] == i:
            x_cell = match_set[match_set['ID'] == CML_data['ID'][current_index]]['X_radar'].item()
            y_cell = match_set[match_set['ID'] == CML_data['ID'][current_index]]['Y_radar'].item()
            #CML_data['TARG_PRCP'][current_index] = radar_set.data[x_cell, y_cell]
            CML_data.at[current_index,'TARG_PRCP'] = radar_set.data[y_cell, x_cell]

            if current_index % 1000 == 0:
                print(current_index, "Timestep:", CML_data['DATE'].iloc[current_index])
            current_index = current_index + 1

            if current_index == len(CML_data):
                break

    return (CML_data, match_set, lon_grid,lat_grid)
