import pandas as pd
import matplotlib.pyplot as plt
main_folder = 'C:/Users/ludod/Documents/MSc Thesis/'
new_small_folder = 'C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/'

sample_small_1link_40ep = 'output_cpu_small_1sam_seql_8_1link_40ep.txt'
sample_small_1link_200ep = 'output_cpu_small_1sam_seql_8_1link_200ep.txt'
sample_small_1link_200ep_lr0008 = 'output_cpu_small_seql_8_1link_200ep_lr0008.txt'
sample_small_1link_50ep_lr0008 = 'output_cpu_small_seql_8_1link_50ep_lr0008.txt'
sample_small_1link_40ep_RDF = 'output_cpu_small_seql_8_1link_40ep_RDF.txt'
sample_small_1link_50ep_RDF = 'output_cpu_small_seql_8_1link_50ep_RDF.txt'
sample_small_1link_200ep_RDF = 'output_cpu_small_seql_8_200ep_RDF.txt'
sample_small_1link_200ep_updatedRDF = 'output_cpu_small_seql_8_200ep_updatedRDF.txt'
sample_small_1link_200ep_updatedRDF_highlr = 'output_cpu_small_seql_8_200ep_updatedRDF_highlr.txt'
sample_small_testaverage = 'output_cpu_small_seql_8_2ep_updatedRDF_highlr.txt'
sample_small_testRDF = 'output_cpu_small_seql_8_100ep_updatedRDF_highlr.txt'
sample_small_newarch = 'output_sql8_numl_4_hids_256_nepo_100_lr00001.txt'
sample_small_newarch_lr = 'output_sql8_numl_4_hids_256_nepo_100_lr00005.txt'
sample_LSTM = 'output_LSTM_seql8_numl_2_hids_256_nepo_100_lr00005.txt'
sample_LSTM_10 = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr00005_median_10links.txt'
sample_LSTM_10_highlr = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr001_median_10links.txt'
sample_notrans_LSTM_10 = 'output_LSTM_seql8_numl_2_hids_256_nepo_10_lr0001_10links_notransform.txt'
sample_50_1ep_RDF = 'output_cpu8_seql_8_50runs_1epochRDF.txt'
sample_onlyrain = 'output_LSTM_seql8_numl_2_hids_128_nepo_100_lr0005_10links_notransform_test.txt'

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

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr
data_small_bestof50 = pd.read_csv(main_folder + sample_50_1ep_RDF)
plt.scatter(data_small_bestof50['outputs_real'],data_small_bestof50['targets_real'])
plt.title('Small dataset seq = 8, meanscaled and with logsine, best of 50 runs, 1 epoch')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr
data_small_seq8_log40ep = pd.read_csv(new_small_folder + sample_small_1link_40ep)
plt.scatter(data_small_seq8_log40ep['outputs'],data_small_seq8_log40ep['targets'])
plt.title('Small dataset seq = 8, meanscaled and with logsine, 40 epochs, 1 link, lr=0.08')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr
data_small_seq8_log200ep = pd.read_csv(new_small_folder + sample_small_1link_200ep)
plt.scatter(data_small_seq8_log200ep['outputs'],data_small_seq8_log200ep['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 200 epochs, 1 link, lr=0.08')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr
data_small_seq8_log200eplr0008 = pd.read_csv(new_small_folder + sample_small_1link_200ep_lr0008)
plt.scatter(data_small_seq8_log200eplr0008['outputs'],data_small_seq8_log200eplr0008['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 200 epochs, 1 link, lr=0.008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 50 ep
data_small_seq8_log50eplr0008 = pd.read_csv(new_small_folder + sample_small_1link_50ep_lr0008)
plt.scatter(data_small_seq8_log50eplr0008['outputs'],data_small_seq8_log50eplr0008['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 50 epochs, 1 link, lr=0.008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 50 ep
data_small_seq8_log40ep_RDF = pd.read_csv(new_small_folder + sample_small_1link_40ep_RDF)
plt.scatter(data_small_seq8_log40ep_RDF['outputs'],data_small_seq8_log40ep_RDF['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 50 epochs, 1 link, lr=0.008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 50 ep
data_small_seq8_log50ep_RDF = pd.read_csv(new_small_folder + sample_small_1link_50ep_RDF)
plt.scatter(data_small_seq8_log50ep_RDF['outputs'],data_small_seq8_log50ep_RDF['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 50 epochs, 1 link, lr=0.008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 200 ep
data_small_seq8_log10ep_updatedRDF = pd.read_csv(new_small_folder + sample_small_1link_200ep_RDF)
plt.scatter(data_small_seq8_log10ep_updatedRDF['outputs'],data_small_seq8_log10ep_updatedRDF['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 80 epochs, 1 link, lr=0.0008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 200 ep
data_small_seq8_log200ep_updatedRDF = pd.read_csv(new_small_folder + sample_small_1link_200ep_updatedRDF)
plt.scatter(data_small_seq8_log200ep_updatedRDF['outputs'],data_small_seq8_log200ep_updatedRDF['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 80 epochs, 1 link, lr=0.0008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 200 ep
data_small_seq8_log200ep_updatedRDF_highlr = pd.read_csv(new_small_folder + sample_small_1link_200ep_updatedRDF_highlr)
plt.scatter(data_small_seq8_log200ep_updatedRDF_highlr['outputs'],data_small_seq8_log200ep_updatedRDF_highlr['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 80 epochs, 1 link, lr=0.0008')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 200 ep
data_small_testaverage = pd.read_csv(new_small_folder + sample_small_testaverage)
plt.scatter(data_small_testaverage['outputs'],data_small_testaverage['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 2 epochs, 1 link, lr=0.0008, new weighted average')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

targets_test = torch.tensor(data_small_testaverage['targets_nontrans'])
outputs_test = torch.tensor(data_small_testaverage['outputs_nontrans'])

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_testRDF = pd.read_csv(new_small_folder + sample_small_testRDF)
plt.scatter(data_small_testRDF['outputs'],data_small_testRDF['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 2 epochs, 1 link, lr=0.001, new weighted average')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_testRDF_newarch = pd.read_csv(new_small_folder + sample_small_newarch)
plt.scatter(data_small_testRDF_newarch['outputs'],data_small_testRDF_newarch['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 2 epochs, 1 link, lr=0.001, new architecture')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_testRDF_newarch_lr = pd.read_csv(new_small_folder + sample_small_newarch_lr)
plt.scatter(data_small_testRDF_newarch_lr['outputs'],data_small_testRDF_newarch_lr['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 2 epochs, 1 link, lr=0.0005, new architecture')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr, 100 ep with early stopping after 20 epochs
data_small_LSTM = pd.read_csv(new_small_folder + sample_LSTM)
plt.scatter(data_small_LSTM['outputs'],data_small_LSTM['targets'])
plt.title('Small dataset seq = 8, meanscaled and \n with logsine, 100 epochs, 1 link, lr=0.0005, LSTM')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

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
