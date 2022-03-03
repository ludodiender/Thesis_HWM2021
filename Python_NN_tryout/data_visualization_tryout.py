import pandas as pd
import matplotlib.pyplot as plt
main_folder = 'C:/Users/ludod/Documents/MSc Thesis/'
new_small_folder = 'C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/'
sample_small = 'output_test_small_1sam_full_withcheckpoint.txt'
sample_large = 'output_cpu_large_1sam_full_withcheckpoint.txt'
sample_large_5 = 'output_cpu_large_5sam_full_higher_lr.txt'
sample_large_seq8 = 'output_cpu_large_1sam_seql_8.txt'
sample_small_seq8 = 'output_cpu_small_1sam_seql_8.txt'
sample_small_seq8_mean = 'output_gpu_small_1sam_seql_8_meanscaled.txt'
sample_small_seq8_mean_logsine = 'output_gpu_small_1sam_seql_8_mstd_logsine.txt'
sample_small_seq8_mlog_10ep = 'output_cpu_small_1sam_seql_8_logsine_10ep.txt' # LR: 1.07e-6
sample_small_seq8_mlog_10ep_highlr = 'output_cpu_small_seql_8_logsine_10ep_highlr.txt' # 0.0022371
sample_small_1link_10ep = 'output_cpu_small_1sam_seql_8_1link.txt'
sample_small_1link_10ep_8lr = 'output_cpu_small_1sam_seql_8_1link_0.08lr.txt'
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

# Plot the target precipitation
plt.hist(testdata_link1['RXMAX'],bins=20)
plt.hist(testdata_link1['RXMIN'],bins=20)
plt.show()

# Plot small dataset, 1 sample
data = pd.read_csv(main_folder+sample_small)
plt.scatter(data['outputs'],data['targets '])
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot large dataset, 1 sample
data_large = pd.read_csv(main_folder+sample_large)
plt.scatter(data_large['outputs'],data_large['targets '])
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot large dataset, best of 5 samples
data_large_5 = pd.read_csv(main_folder + sample_large_5)
plt.scatter(data_large_5['outputs'],data_large_5['targets '])
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot large dataset, sequence length 8 1 sample
data_large_seq8 = pd.read_csv(main_folder + sample_large_seq8)
plt.scatter(data_large_seq8['outputs'],data_large_seq8['targets '])
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample
data_small_seq8_mean = pd.read_csv(main_folder + sample_small_seq8_mean)
plt.scatter(data_small_seq8_mean['outputs'],data_small_seq8_mean['targets'])
plt.title('Small dataset seq = 8, scaled by mean')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled
data_small_seq8_log = pd.read_csv(main_folder + sample_small_seq8_mean_logsine)
plt.scatter(data_small_seq8_log['outputs'],data_small_seq8_log['targets'])
plt.title('Small dataset seq = 8, meanscaled and with logsine')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled
data_small_seq8_log10 = pd.read_csv(main_folder + sample_small_seq8_mlog_10ep)
plt.scatter(data_small_seq8_log10['output_nontrans'],data_small_seq8_log10['targets_nontrans '])
plt.title('Small dataset seq = 8, meanscaled and with logsine, 10 epochs')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled
data_small_seq8_log10_highlr = pd.read_csv(main_folder + sample_small_seq8_mlog_10ep_highlr)
plt.scatter(data_small_seq8_log10_highlr['outputs'],data_small_seq8_log10_highlr['targets'])
plt.title('Small dataset seq = 8, meanscaled and with logsine, 10 epochs, higher lr')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link
data_small_seq8_log10_1link = pd.read_csv(new_small_folder + sample_small_1link_10ep)
plt.scatter(data_small_seq8_log10_1link['outputs'],data_small_seq8_log10_1link['targets'])
plt.title('Small dataset seq = 8, meanscaled and with logsine, 10 epochs, 1 link')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.show()

# Plot small dataset, sequence lenght 8 1 sample mean scaled 1 link higher lr
data_small_seq8_log10_1link_lr = pd.read_csv(new_small_folder + sample_small_1link_10ep_8lr)
plt.scatter(data_small_seq8_log10_1link_lr['outputs'],data_small_seq8_log10_1link_lr['targets'])
plt.title('Small dataset seq = 8, meanscaled and with logsine, 10 epochs, 1 link, lr=0.08')
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
