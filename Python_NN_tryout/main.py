# This is a sample Python script.

# First example python script. The script allows to switch between the locally stored complete dataset (not publicly available)
# and a sample dataset of one day that can be used to try out the script.
# TODO: implement switch between local dataset on my PC and a smaller example dataset


import data_functions as df
import numpy as np
import matplotlib.pyplot as plt

data = df.load_CML_data('20110714', 'NOKIA')
data_with_target, match_set = df.ReadRainLocation(data)
train,test,val = df.split_data(data_with_target,40,40,20)
error_data = data_with_target[data_with_target['TARG_PRCP'] == 65535.00]

# CODE NOT NEEDED
target_data, radar_file, coord_grid = df.load_radar_data_h5(20110628, 1145)
target_data_nc_prcp,target_data_nc = df.load_radar_data_netcdf(20110712,1145)

# CoorSystemInputData <- "+init=epsg:4326"	# WGS84

plt.imshow(target_data_nc_prcp, vmin=0, vmax=0.05)
plt.colorbar()
plt.show()


