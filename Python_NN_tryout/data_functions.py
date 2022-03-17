import pandas as pd
import numpy as np
import h5py
import os
import netCDF4 as nc
import time
from os import listdir
from os.path import isfile, join
from datetime import datetime
import pytz


def load_CML_data(provider, date_selected = None, filename=None):

    # This function loads the specific CML data from a specified day. It takes two variables
    # - date_selected (string): the date to be extracted from the dataset, format YYYYMMDD
    # - provider (string): the name of the data provider (NOKIA or NEC possible)
    if (date_selected == None) & (filename==None):
        raise TypeError('Specify either a selected date or a filename')
    if (date_selected != None) & (filename !=None):
        raise TypeError('Specify either a selected date or a filename, not both')

    header_names = ['SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON', 'FREQ', 'DATE', 'RXMIN', 'RXMAX',
                    'DIST', 'P4_start.v', 'P4_start.u', 'P4_end.v', 'P4_end.u', 'PROV']
    if filename == None:
        # Create the right folder URL for the right dataset and specify the headers
        folder_URL_NEC = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NEC_data/NEC_data/'
        folder_URL_NOKIA = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NOKIA_data/NOKIA_data/'


        # Select the right data location based on the provider and the date_selected
        if provider == 'NOKIA':
            spec_URL = folder_URL_NOKIA + 'nokia_fh_' + date_selected + '.txt'
        elif provider == 'NEC':
            spec_URL = folder_URL_NEC + 'NEC_' + date_selected + '.txt'
    else:
        spec_URL = filename

    # Load the dataset
    data = pd.read_csv(spec_URL, header=None, delim_whitespace=True, names=header_names)
    return data


def create_ID_table(dataset,coord_from_harddrive):
    data_copy = dataset.copy()
    data_copy.insert(0, "ID", value=np.nan)
    # Add average Longitude and Latitude for the middle of the link
    avg_lon = (dataset['SITE_LON'] + dataset['FAR_LON']) / 2
    avg_lat = (dataset['SITE_LAT'] + dataset['FAR_LAT']) / 2
    data_copy.loc[:,"AVG_LON"] = avg_lon
    data_copy.loc[:,"AVG_LAT"] = avg_lat

    groupIDS = data_copy.groupby(['SITE_LON','SITE_LAT','FAR_LON','FAR_LAT','FREQ']).ngroup() + 1

    data_copy.loc[:, 'ID'] = groupIDS  # Start ID at 1, by adding +1 to the groupby function

    matching_ID_table = data_copy[['ID','AVG_LON','AVG_LAT','FREQ']].drop_duplicates(ignore_index=True)
    if coord_from_harddrive:
        coord_path = 'D:/radarcoordinaten_NL25_1km2_WGS84_full.dat'
    else:
        coord_path = 'C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/radarcoordinaten_NL25_1km2_WGS84_full.dat'
    coord_grid = pd.read_csv(coord_path)
    #print('Coordinates loaded')
    # The coordinates for each row of the dataset change slightly from row to row as the Earth curves. The spread in
    # these coordinates however is smaller than the difference from column to column, so an average is taken for each grid cell.
    # This ensures that the right column and row are selected for the right CML

    lon_grid = coord_grid['lon'].values.reshape(765, 700).mean(axis=0)
    lat_grid = coord_grid['lat'].values.reshape(765, 700).mean(axis=1)

    # grid cell in the radar image.
    matching_ID_table['X_radar'] = 0
    matching_ID_table['Y_radar'] = 0

    # Calculate the distance to all rows and columns and take the smallest of these distances.
    for i in range(0, len(matching_ID_table)):
        Dist_lon = np.sqrt(np.power((lon_grid - matching_ID_table['AVG_LON'][i]), 2))
        Dist_lat = np.sqrt(np.power((lat_grid - matching_ID_table['AVG_LAT'][i]), 2))
        matching_ID_table.at[i, 'X_radar'] = Dist_lon.argmin()
        matching_ID_table.at[i, 'Y_radar'] = Dist_lat.argmin()

    return matching_ID_table, lon_grid, lat_grid

def generate_ID(dataset, matchset, lon_grid, lat_grid):
    dataset.insert(0, "ID", value=np.nan)
    # Add average Longitude and Latitude for the middle of the link
    dataset['AVG_LON'] = (dataset['FAR_LON'] + dataset['SITE_LON']) / 2
    dataset['AVG_LAT'] = (dataset['FAR_LAT'] + dataset['SITE_LAT']) / 2

    for i in range(0,len(dataset)):
        match_index = matchset.index[(matchset['AVG_LON'] == dataset['AVG_LON'][i]) &
                                     (matchset['AVG_LAT'] == dataset['AVG_LAT'][i]) &
                                     (matchset['FREQ']    == dataset['FREQ'][i])].tolist()
        if len(match_index) == 0: # If the link is not yet in the matchset, add it with a new ID
            maxID = np.max(matchset['ID'])
            print(len(matchset))
            matchset = matchset.append({'ID': maxID+1, 'AVG_LON': dataset['AVG_LON'][i], 'AVG_LAT': dataset['AVG_LAT'][i], 'FREQ': dataset['FREQ'][i]}, ignore_index = True)
            match_index = [len(matchset)-1]
            print(match_index,len(matchset))
            # Calculate the closest point on the grid and add it to the matchset
            #Dist_lon = np.sqrt(np.power((lon_grid - matchset['AVG_LON'][match_index]), 2))
            #Dist_lat = np.sqrt(np.power((lat_grid - matchset['AVG_LAT'][match_index]), 2))

            Dist_lon = np.sqrt(np.power((lon_grid - matchset.at[match_index[0], 'AVG_LON']), 2))
            Dist_lat = np.sqrt(np.power((lat_grid - matchset.at[match_index[0], 'AVG_LAT']), 2))

            matchset.at[match_index, 'X_radar'] = Dist_lon.argmin()
            matchset.at[match_index, 'Y_radar'] = Dist_lat.argmin()

        dataset.at[i,'ID'] = int(matchset.at[match_index[0],'ID'].item()) # Select the first ID if there is something wrong with selecting ID
    return dataset, matchset

def write_to_csv(dataset, provider,data_model_function):
    t0 = time.time()
    dataset_sorted = dataset.sort_values(['ID','DATE'])

    # Temporary file path
    header_names = ['ID','SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON', 'FREQ', 'DATE', 'RXMIN', 'RXMAX',
                    'DIST', 'P4_start.v', 'P4_start.u', 'P4_end.v', 'P4_end.u', 'PROV','AVG_LON','AVG_LAT','TARG_PRCP']
    if data_model_function == 'train':
        folderpath = 'D:/CML_RAD_perID_4months_UTC/2011_training/' + provider + '/'
    elif data_model_function == 'test':
        folderpath = 'D:/CML_RAD_perID_4months_UTC/2012_testing/' + provider + '/'
    elif data_model_function == 'validate':
        folderpath = 'D:/CML_RAD_perID_4months_UTC/2013_validating/' + provider + '/'
    # folderpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_per_ID/'
    nr_of_IDS = len(dataset_sorted['ID'].unique())
    ID_cumsum = dataset_sorted['ID'].value_counts().cumsum()
    occ_of_IDS = pd.concat([pd.Series([0]),ID_cumsum])
    for i in range(0,nr_of_IDS):
        measure_sample = dataset_sorted.iloc[occ_of_IDS.iloc[i]:occ_of_IDS.iloc[i+1],:]
        linkID = int(measure_sample.iloc[0,:]['ID'].item())
        filepath = folderpath + provider + '_linkID_' + str(linkID) +'.txt'
        if os.path.exists(filepath):
            measure_sample.to_csv(filepath, mode='a',index=False, header = False)
        else:
            measure_sample.to_csv(filepath, mode='w',index=False, header=header_names)


def split_data(dataset, train_perc, test_perc, val_perc=0):
    """ This function splits the supplied data in three different sets (or two if val_perc is not supplied). The data are sorted
    by ID and Date before returned

    :param dataset (pd.dataframe): dataset to be split
    :param train_perc (int): percentage of the dataset that will be used for training
    :param test_perc (int): percentage of the dataset that will be used for testing
    :param val_perc (int): percentage of the dataset that will be used for validating
    :return: three separate pd.dataframes for training, testing and validating
    """

    # TODO: make this function compatible with numpy arrays, add multiple strategies of splitting the data (spatial or temporal)

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


def load_radar_data_netcdf(date_selected, time_selected, from_ext_drive = False):
    # This function loads the radar data from a specific time in a netCDF format.
    # - date_selected (int): selected date
    # - time_selected (int): selected time
    endyear = str(date_selected)[0:4] # Year ends on the last date of the year
    startyear = str(int(endyear) - 1)  # Year starts on the final date of the previous year

    if (str(date_selected)[4:] == '0101') & (str(time_selected).zfill(4)=='0000'):
        endyear_folder = str(int(endyear)-1)
        startyear_folder = str(int(startyear)-1)
    else:
        endyear_folder = endyear
        startyear_folder = startyear

    if from_ext_drive:
        base_URL = 'D:/RADNL_CLIM_EM_MFBSNL25_05m_'+startyear_folder+'1231T235500_'+endyear_folder+'1231T235500_netCDF4_0002/'+endyear_folder+'/'
    else:
        base_URL = 'C:/Users/ludod/Documents/MSc Thesis/RADNL_CLIM_EM_MFBSNL25_05m_'+startyear_folder+'1231T235500_'+endyear_folder+'1231T235500_netCDF4_0002/'+endyear_folder+'/'

    # Select the month from the selected date to go to the right folder
    month = str(date_selected)[4:6]

    # The data needs to be aggregated to 15 minutes.
    for i in [1, 2, 3]:
        if time_selected == 0000:  # TODO: fix this part, difficult to deal with the string and integer formats
            date_selected = date_selected - 1
            if date_selected % 100 == 0: # Start of the month
                if int(str(date_selected)[4:]) in [200, 400, 600, 800, 900, 1100]: date_selected = date_selected - 69
                elif int(str(date_selected)[4:]) in [500, 700, 1000, 1200]: date_selected = date_selected - 70
                elif int(str(date_selected)[4:]) in [100]:
                    date_selected = date_selected - 8869
                    month = '13' # Set to 13, 1 will be subtracted later to fix
                elif int(str(date_selected)[0:4]) == 2012: date_selected = date_selected - 71
                else:
                    date_selected = date_selected - 72

                month = str(int(month)-1).zfill(2)


            time_selected = time_selected + 2400
        if str(time_selected).zfill(4)[2:4] == '00':
            time_selected_prior = time_selected - 40 - i * 5
        else:
            time_selected_prior = time_selected - i * 5

        spec_URL = base_URL + month + '/RAD_NL25_RAC_MFBS_EM_5min_' + str(date_selected) + str(time_selected_prior).zfill(4) + '.nc'
        temp_dataset = nc.Dataset(spec_URL)
        temp_precp = temp_dataset['image1_image_data'][0, :, :]

        if i == 1:
            total_radar = temp_precp
        else:
            total_radar = total_radar + temp_precp

    return (total_radar, temp_dataset)


def ReadRainLocation(CML_data,matchset_XY, from_ext_drive):
    # This function takes the CML data with time and AVG_LON and AVG_LAT of the link and returns a dataset with an extra column
    # that represents the true value for the precipitation.
    # TODO: Don't use the middle of the link but a weighted average of all pixels it passes through.

    # Load the grid with the coordinates for the radar.
    t0 = time.time()
    cwd = os.getcwd()

    # Create a 'unique' list of times in the dataset
    times_inCML = CML_data['DATE'].unique()
    times_inCML.sort()


    # Create a list of UTC times associated to these times to match with the radar
    UTC_times_inCML = times_inCML.copy()
    format = '%Y%m%d%H%M'
    for i in range(len(times_inCML)):
        local_time_i = datetime.strptime(str(times_inCML[i]),format)
        utc_time_i = local_time_i.astimezone(pytz.UTC)
        str_utc_time = utc_time_i.strftime(format)
        UTC_times_inCML[i] = str_utc_time

    current_index = 0  # Keep track of where you are in the list
    CML_data['TARG_PRCP'] = 0.0000  # Create an empty column to be filled later
    CML_data = CML_data.sort_values(by='DATE')

    # Loop over all timesteps in the file
    for i in range(len(times_inCML)):

        radar_set, _ = load_radar_data_netcdf(int(str(UTC_times_inCML[i])[0:8]), int(str(UTC_times_inCML[i])[8:12]),from_ext_drive=from_ext_drive)


        # The while loop ensures that the for loop continues to the next time step as all links have been added
        while CML_data['DATE'].iloc[current_index] == times_inCML[i]:
            x_cell = int(matchset_XY[matchset_XY['ID'] == CML_data.loc[current_index, 'ID']]['X_radar'].item())
            y_cell = int(matchset_XY[matchset_XY['ID'] == CML_data.loc[current_index, 'ID']]['Y_radar'].item())

            CML_data.at[current_index,'TARG_PRCP'] = radar_set.data[y_cell, x_cell]

            #if current_index % 10000 == 0:
                #print('Target added for sample',current_index, 'out of ',len(CML_data))
            current_index = current_index + 1

            if current_index == len(CML_data):
                break
    CML_data['TARG_PRCP'] = CML_data['TARG_PRCP'] * 4 # To convert from mm/15 min to mm/h
    #print(time.time()-t0)

    return CML_data


def filter_data_error(data_target):
    # Remove the datapoints where no radar data is available
    data_with_target_errorless = data_target[data_target['TARG_PRCP'] != (4*65535.00)]

    # Filter out the erroneous CML data values (-1.0)
    data_with_target_errorless = data_with_target_errorless[data_with_target_errorless['RXMIN'] != -1.0]
    data_with_target_errorless = data_with_target_errorless[data_with_target_errorless['RXMAX'] != -1.0]

    # Remove Double ID's (different polarization but unable to split the two, averaging could be done)
    ID_values = data_with_target_errorless['ID'].value_counts()  # Get the number of times a certain ID occurs

    # Select the number of occurances that occurs the most
    max_ID_length = ID_values.value_counts()[ID_values.value_counts() == ID_values.value_counts().max()].index.values[0]
    IDs_tokeep = ID_values[ID_values == max_ID_length].index.to_list()  # Add all IDs that have the highest occurring occurence to a list

    data_with_target_errorless = data_with_target_errorless[
        data_with_target_errorless['ID'].isin(IDs_tokeep)]  # Select only those IDs that occur in the list

    return data_with_target_errorless


def get_ID_table_from_data(foldername):
    headers = ['ID', 'SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON', 'FREQ', 'DATE', 'RXMIN', 'RXMAX', 'DIST',
               'P4_start.v', 'P4_start.u', 'P4_end.v', 'P4_end.u', 'PROV', 'AVG_LON', 'AVG_LAT', 'TARG_PRCP']
    empty_matchset = pd.DataFrame(columns=headers)
    files = [f for f in listdir(foldername) if isfile(join(foldername, f))]

    for f in files:

        f_full = foldername + '/' + f
        data_sup = pd.read_csv(f_full, header=0)
        j = 0
        for i in range(0, len(data_sup)):
            j += 1
            if j == 1:
                data_line = data_sup.iloc[j]
                empty_matchset = empty_matchset.append(data_line)
                break
        print('Done with', f)

    return(empty_matchset)
'''
matching_ID_table = empty_matchset_new2[['ID','AVG_LON','AVG_LAT','FREQ']].reset_index(drop=True)
matching_ID_table = matching_ID_table.assign(X_radar = 0)
matching_ID_table = matching_ID_table.assign(Y_radar = 0)
# Calculate the distance to all rows and columns and take the smallest of these distances.
for i in range(0, len(matching_ID_table)):
    Dist_lon = np.sqrt(np.power((lon_grid - matching_ID_table.iloc[i]['AVG_LON']), 2))
    Dist_lat = np.sqrt(np.power((lat_grid - matching_ID_table.iloc[i]['AVG_LAT']), 2))
    matching_ID_table.at[i, 'X_radar'] = Dist_lon.argmin()
    matching_ID_table.at[i, 'Y_radar'] = Dist_lat.argmin()
'''


######################################
## Full data to hard drive function ##
######################################
def data_to_harddrive(filepath,provider,longrid=None,latgrid=None,match_table = None):
    t0 = time.time()
    # Get a list of all the files in the filepath
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    # For every file, run the whole script of combining it with the right file somewhere on the disk
    count=0
    for f in files:
        t0_0 = time.time()
        filename = filepath+'/'+f
        year = f[9:13] # Extract the year from the file name to send the results to the right folder
        month = f[13:15]

        data = load_CML_data(provider, filename=filename)

        if data.loc[0, 'PROV'].endswith(';'):
            data['PROV'] = data['PROV'].str.slice(start=0, stop=-1)
        if year == '2011': model_function = 'train'
        if year == '2012': model_function = 'test'
        if year == '2013': model_function = 'validate'

        if count == 0:
            if isinstance(match_table,type(None)):
                data_ID = data[:]
                matching_ID_table, lon_grid, lat_grid = create_ID_table(data_ID, coord_from_harddrive=True)
            else:
                matching_ID_table = match_table
                lon_grid = longrid
                lat_grid = latgrid


        data, matching_ID_table = generate_ID(data, matching_ID_table, lon_grid, lat_grid)
        print('IDs generated for file ', f)
        data_with_target = ReadRainLocation(data, matching_ID_table, from_ext_drive=True)
        data_with_target_errorless = filter_data_error(data_with_target)

        write_to_csv(data_with_target_errorless, provider, data_model_function=model_function)

        count += 1
        print('Done with file:', f, '. Time spent:', time.time() - t0_0)

    print('Done with all files. Total time spent:', time.time()-t0)

    # For 2011 -> training, 2012 -> testing, 2013 -> validating


#### TRY OUT WITH DATETIME
from datetime import datetime
import pytz
format = '%Y%m%d%H%M'
date = '201104010815'

local_time = datetime.strptime(date,format)
utc_time = local_time.astimezone(pytz.UTC)