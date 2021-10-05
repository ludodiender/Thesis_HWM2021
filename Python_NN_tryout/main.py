# This is a sample Python script.

# First example python script. The script allows to switch between the locally stored complete dataset (not publicly available)
# and a sample dataset of one day that can be used to try out the script.
# TODO: implement switch between local dataset on my PC and a smaller example dataset


import data_functions
import numpy as np

data = data_functions.load_data('20110628', 'NOKIA')
train,test,val = data_functions.split_data(data,40,40,20)



