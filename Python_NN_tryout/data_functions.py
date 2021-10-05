import pandas as pd
import numpy as np

def load_data(date_selected, provider):
    # TODO: add functionality of selecting multiple days at once (will take a lot of time!)
    # TODO: add compatibility with numpy arrays (PyTorch package needs numpy arrays for tensors)
    # This function loads the specific CML data from a specified day. It takes two variables
    # - date_selected (string): the date to be extracted from the dataset
    # - provider (string): the name of the data provider (NOKIA or NEC possible)

    # Create the right folder URL for the right dataset and specify the headers
    folder_URL_NEC = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NEC_data/NEC_data/'
    folder_URL_NOKIA = 'C:/Users/ludod/Documents/MSc Thesis/transfer_1153027_files_feb7e0a7/NOKIA_data/NOKIA_data/'
    header_names = ['SITE_LAT','SITE_LON','FAR_LAT','FAR_LON','FREQ','DATE','RXMIN','RXMAX',
                    'DIST','P4_start.v','P4_start.u','P4_end.v','P4_end.u','PROV']

    # Select the right data location based on the provider and the date_selected
    if provider == 'NOKIA': spec_URL = folder_URL_NOKIA +'nokia_fh_'+date_selected+'.txt'
    elif provider == 'NEC': spec_URL = folder_URL_NEC+'NEC_'+date_selected+'.txt'

    # Load the dataset
    data = pd.read_csv(spec_URL, header = None, delim_whitespace = True, names = header_names)

    # Add a unique identifier for each link in the dataset
    data.insert(0, "ID", value=np.nan)
    data['ID'] = data.groupby(['SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON']).ngroup() + 1 # Start ID at 1, by adding +1 to the groupby function

    return(data)

def split_data(data, train_perc, test_perc, val_perc = 0):
    # TODO: make this function compatible with numpy arrays, add multiple strategies of splitting the data (spatial or temporal)

    # This function splits the supplied data in three different sets (or two if val_perc is not supplied) and returns those
    # It takes four arguments
    # - data (dataframe): dataset to be split
    # - train_perc (int): percentage of the dataset that will be used for training
    # - test_perc (int): percentage of the dataset that will be used for testing
    # - val_perc (int): percentage of the dataset that will be used for validating

    # Check if the percentages match and make sure that none of the percentages are negative
    if train_perc + test_perc + val_perc != 100:
        raise ValueError('The percentages do not add up to 100%. Check your numbers again')
    elif (train_perc or test_perc or val_perc) <0:
        raise ValueError('Insert only positive integers as percentages')
    else:
        # Split the data here
        train_idx = round(len(data) * (train_perc / 100))
        test_idx = round(len(data) * (test_perc / 100)) + train_idx
        train_data = data.loc[0:train_idx, :]
        test_data  = data.loc[train_idx:test_idx,:]
        val_data   = data.loc[test_idx:,:]

    return(train_data,test_data,val_data)



