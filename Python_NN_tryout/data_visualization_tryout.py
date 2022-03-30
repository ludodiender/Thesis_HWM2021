import pandas as pd
import matplotlib.pyplot as plt
main_folder = 'C:/Users/ludod/Documents/MSc Thesis/'
new_small_folder = 'C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/'

sample_LSTM = 'output_LSTM_seql8_numl_2_hids_256_nepo_100_lr00005.txt'
sample_LSTM_10 = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr00005_median_10links.txt'
sample_LSTM_10_highlr = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr001_median_10links.txt'
sample_notrans_LSTM_10 = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr0001_10links_notransform.txt'
sample_50_1ep_RDF = 'output_cpu8_seql_8_50runs_1epochRDF.txt'
sample_onlyrain = 'output_LSTM_seql8_numl_2_hids_128_nepo_100_lr0005_10links_notransform_test.txt'
sample_10links_notransform_UTC = 'output_LSTM_UTC_numl_2_hids_128_nepo_10_lr0001_10link.txt'
sample_10links_transform_UTC = 'output_LSTM_UTC_numl_2_hids_128_nepo_10_lr0001_10link_transformed.txt'
sample_10links_transform_UTC_median = 'output_LSTM_UTC_numl_2_hids_128_nepo_10_lr0001_10link_transformed_median.txt'
sample_10links_notransform_UTC_onlyrain = 'output_LSTM_UTC_numl_2_hids_128_nepo_10_lr00005_10link_nontransformed_onlyrain.txt'
sample_test = 'output_LSTM_MSE_test.txt'
sample_test_trans = 'output_LSTM_MSE_test_trans.txt'
sample_test_trans_RDF = 'output_LSTM_RDF_test_trans.txt'
sample_test_trans_MAE_RDF = 'output_LSTM_RDF_MAE_test_trans.txt'
sample_test_full_features_RDF = 'output_LSTM_RDF_test_trans_features.txt'
# Plot the target precipitation
testlink = pd.read_csv('C:/Users/ludod/Documents/MSc Thesis/CML_Small_testset/train/NOKIA_linkID_1003.txt', sep=',',header=0,names=header_names)
RXMIN_med = np.median(testlink['RXMIN'])
RXMAX_med = np.median(testlink['RXMAX'])

figure1 = plt.figure()
ax = figure1.add_subplot()
c = ax.scatter(testlink['RXMIN'] - RXMIN_med, testlink['TARG_PRCP'])
ax.set_title('Relation between Minimal RSL and precipitation')
ax.set_xlabel('Minimum RSL - median [dBm]')
ax.set_ylabel('Observed precipitation [mm/h]')

ax2 = figure1.add_subplot()
b = ax2.scatter(testlink['RXMAX'] - RXMAX_med, testlink['TARG_PRCP'])
ax2.set_title('Relation between Minimal RSL and precipitation')
ax2.set_xlabel('Minimum RSL - median [dBm]')
ax2.set_ylabel('Observed precipitation [mm/h]')
figure1.show()


plt.scatter(testlink['RXMAX'] - RXMAX_med, testlink['TARG_PRCP'])
plt.title('Relation between Maximum RSL and precipitation')
plt.xlabel('Maximum RSL - median [dBm]')
plt.ylabel('Observed precipitation [mm/h]')
plt.show()

plt.scatter(testlink['RXMAX'][4:] - RXMAX_med, testlink['TARG_PRCP'][:-4])
plt.title('Relation between Maximum RSL and precipitation SHIFTED')
plt.xlabel('Maximum RSL - median [dBm]')
plt.ylabel('Observed precipitation [mm/h]')
plt.show()

plt.scatter(testlink['RXMIN'][4:] - RXMIN_med, testlink['TARG_PRCP'][:-4])
plt.title('Relation between Minimum RSL and precipitation SHIFTED')
plt.xlabel('Minimum RSL - median [dBm]')
plt.ylabel('Observed precipitation [mm/h]')
plt.show()

figure,ax1 = plt.subplots()
xdata = testlink['DATE'][19280:19369]
ax1.plot(xdata,testlink['RXMIN'][19280:19369] - RXMIN_med)
ax2 = ax1.twinx(color='tab:red')
ax1.plot(xdata,testlink['RXMAX'][19280:19369] - RXMAX_med, color='tab:green')
ax2.plot(xdata,testlink['TARG_PRCP'][19280:19369], color='tab:red')
plt.title('Time series for precipitation and received signal level')
ax1.set_xlabel('Date [YYYYMMDDHHMM]')
ax2.set_ylabel('Rainfall rate [mm/h]',color='tab:red')
ax1.set_ylabel('Minimum received signal level attenuation [dBm]')
#matplotlib.dates.AutoDateLocator()
#ax1.xaxis.set_ticks((xdata.iloc[0],xdata.iloc[20],xdata.iloc[40],xdata.iloc[60],xdata.iloc[80]))
ax1.set_xticklabels(labels=(xdata.iloc[0],xdata.iloc[20],xdata.iloc[40],xdata.iloc[60],xdata.iloc[80]),rotation=15)
figure.show()



plt.hist(testlink1['RXMAX'],bins=20)
plt.hist(testdata_link1['RXMIN'],bins=20)
plt.show()






targets_test = torch.tensor(data_small_testaverage['targets_nontrans'])
outputs_test = torch.tensor(data_small_testaverage['outputs_nontrans'])

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_LSTM_10 = pd.read_csv(new_small_folder + sample_LSTM_10)
plt.scatter(data_small_LSTM_10['outputs'],data_small_LSTM_10['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 100 epochs, 10 links, lr=0.001, LSTM')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_LSTM_10_highlr = pd.read_csv(new_small_folder + sample_LSTM_10_highlr)
plt.scatter(data_small_LSTM_10_highlr['outputs'],data_small_LSTM_10_highlr['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 100 epochs, 10 links, lr=0.001, LSTM')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_notrans_LSTM_10_highlr = pd.read_csv(new_small_folder + sample_notrans_LSTM_10)
plt.scatter(data_small_notrans_LSTM_10_highlr['outputs_nontrans'],data_small_notrans_LSTM_10_highlr['targets_nontrans'])
plt.title('Small dataset, not scaled or transformed, 10 links LSTM')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

plt.hist2d(data_small_notrans_LSTM_10_highlr['outputs_nontrans'],data_small_notrans_LSTM_10_highlr['targets_nontrans'], bins=(5000,10), cmap=plt.cm.jet)
plt.show()
# Plot small dataset, only rain events and without scaling or transforming.
data_small_onlyrain = pd.read_csv(new_small_folder + sample_onlyrain)
plt.scatter(data_small_onlyrain['outputs_nontrans'],data_small_onlyrain['targets_nontrans'])
plt.title('Small dataset, not scaled or transformed, only rain events included. 10 links LSTM')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

plt.hist2d(data_small_onlyrain['outputs_nontrans'],data_small_onlyrain['targets_nontrans'], bins=(10,50), cmap=plt.cm.jet)
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_notrans = pd.read_csv(new_small_folder + sample_10links_notransform_UTC)
plt.scatter(data_small_UTC_notrans['outputs_nontrans'],data_small_UTC_notrans['targets_nontrans'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_trans = pd.read_csv(new_small_folder + sample_10links_transform_UTC)
plt.scatter(data_small_UTC_trans['outputs'],data_small_UTC_trans['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs, transformed')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_trans_median = pd.read_csv(new_small_folder + sample_10links_transform_UTC_median)
plt.scatter(data_small_UTC_trans_median['outputs'],data_small_UTC_trans_median['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n transformed and features scaled with median')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_notrans_onlyrain = pd.read_csv(new_small_folder + sample_10links_notransform_UTC_onlyrain)
plt.scatter(data_small_UTC_notrans_onlyrain['outputs_nontrans'],data_small_UTC_notrans_onlyrain['targets_nontrans'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n non transformed and features scaled with median')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_test = pd.read_csv(new_small_folder + sample_test)
plt.scatter(data_small_UTC_test['outputs_nontrans'],data_small_UTC_test['targets_nontrans'])
plt.title('LSTM, 10 links. Correct timings (UTC), 1 epochs,\n non-transformed and MSE as loss function')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_test_trans = pd.read_csv(new_small_folder + sample_test_trans)
plt.scatter(data_small_UTC_test_trans['outputs'],data_small_UTC_test_trans['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n transformed and MSE as loss function')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, only rain events and without scaling or transforming.
data_small_UTC_test_trans_RDF = pd.read_csv(new_small_folder + sample_test_trans_RDF)
plt.scatter(data_small_UTC_test_trans_RDF['outputs'],data_small_UTC_test_trans_RDF['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n transformed and RDF MSE as loss function')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()


# Plot small dataset, all events with scaling, MAE RDF as loss
data_small_UTC_test_trans_MAE_RDF = pd.read_csv(new_small_folder + sample_test_trans_MAE_RDF)
plt.scatter(data_small_UTC_test_trans_MAE_RDF['outputs'],data_small_UTC_test_trans_MAE_RDF['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n transformed and RDF MAE as loss function')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, all events with scaling, MAE RDF as loss
data_small_UTC_test_full_features = pd.read_csv(new_small_folder + sample_test_full_features_RDF)
plt.scatter(data_small_UTC_test_full_features['outputs'],data_small_UTC_test_full_features['targets'])
plt.title('LSTM, 10 links. Correct timings (UTC), 10 epochs,\n transformed and RDF MSE as loss function, full data')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()
#### PLOT RDF ############
plt.plot(testx,RDF(torch.tensor(testx)))
plt.xlabel('Precipitation (mm/h)')
plt.ylabel('Rain Distribution Factor')
plt.title('Rain Distribution Factor')
plt.show()
intermediate_data_large5_16epoch = pd.read_csv(main_folder + 'progress_39154824_5samples_0003_16epoch.csv')
intermediate_data_large5_13epoch = pd.read_csv(main_folder + 'progress_39154824_5samples_0001_13epoch.csv')
intermediate_data_large5_6epoch = pd.read_csv(main_folder + 'progress_39154824_5samples_0002_6epoch.csv')
intermediate_data_large5_win = pd.read_csv(main_folder + 'progress_39154824_5samples_0000_win.csv')
intermediate_data_large5_1epoch = pd.read_csv(main_folder + 'progress_39154824_5samples_0004_1epoch.csv')


plt.plot(intermediate_data_large5_16epoch['loss_trans'])
plt.plot(intermediate_data_large5_13epoch['loss_trans'])
plt.plot(intermediate_data_large5_6epoch['loss_trans'])
plt.plot(intermediate_data_large5_1epoch['loss_trans'], 'o')
plt.plot(intermediate_data_large5_win['loss_trans'], 'o')
plt.xlabel('Epochs')
plt.ylabel('Test loss [mm/h]')
plt.title('Test loss per epoch of five different hyperparameter samples')
plt.legend(['16 epochs','13 epochs','6 epochs','1 epoch 2 layers','1 epoch 16 layers'])
plt.show()
