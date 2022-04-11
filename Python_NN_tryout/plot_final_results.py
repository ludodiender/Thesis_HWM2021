# Script with a few functions that visualize the data that I have
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir

def plt_model_observed_CV_r2(filename: str, fontsize: int):
    dataset = pd.read_csv(filename)
    f = plt.figure()
    ax = f.add_subplot(111)
    outputs = dataset['outputs']
    targets = dataset['targets']
    # Calculate the statistics for the plot and add these to the plot
    cv = np.std(outputs) / np.mean(outputs)
    r2 = np.square(np.corrcoef(outputs, targets)[0, 1])
    RMSE= np.sqrt(np.mean(np.square((outputs - targets))))
    Cv_str = 'CV: ' + str(round(cv, 4))
    r2_str = '$\mathregular{R^{2}: }$' + str(round(r2, 3))
    RMSE_str = 'RMSE: ' + str(round(RMSE,4))

    plt.hexbin(outputs, targets, cmap='copper_r',gridsize=50, mincnt=1)
    #plt.scatter(outputs,targets,s=4)
    plt.title('Predicted vs. observed 15-min rainfall intensities')
    plt.xlabel('Model prediction [mm/h]', fontsize=fontsize)
    plt.ylabel('Observation [mm/h]', fontsize=fontsize)
    plt.tick_params(axis='x',labelsize = fontsize-2)
    plt.tick_params(axis='y', labelsize= fontsize - 2)
    plt.text(0.85, 0.9, Cv_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.85, 0.95, r2_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.85,0.85,RMSE_str,ha='center',va='center',transform=ax.transAxes)
    cb = plt.colorbar(label='Occurence [-]')
    cb.set_label(label='Occurence [-]', size=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/scatter_final.png')
    plt.show()

plt_model_observed_CV_r2('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout//output_LSTM_RDF_test_trans_features.txt', fontsize=14)


def plot_validationset_freq_dist(colormap, fontsize = 16):
    valset = pd.read_csv('C:/Users/ludod/Documents/MSc Thesis/val_table_full.txt', header=0)
    fig = plt.figure(figsize=(8,14))
    plt.subplots_adjust(wspace=0.25,hspace=0.25)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,(3,4))

    freq = valset['FREQ']
    dist = valset['DIST']

    # First graph for the frequency distribution
    ax1.hist(freq, bins=20,color='dimgrey')
    ax1.set_xlabel('Frequency [GHz]', fontsize=fontsize)
    ax1.set_ylabel('Occurence [-]', fontsize = fontsize)
    ax1.tick_params(axis='x', labelsize= fontsize-2)
    ax1.tick_params(axis='y', labelsize=fontsize - 2)
    ax1.text(0.945,0.95, 'A',ha='center',va='center', transform=ax1.transAxes, weight='bold', fontsize=fontsize)

    # Second graph for the link path length distribution
    ax2.hist(dist, bins=30, color='dimgrey')
    ax2.set_xlabel('Link path length [km]', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize - 2)
    ax2.tick_params(axis='y', labelsize=fontsize - 2)
    ax2.text(0.945, 0.95, 'B', ha='center', va='center', transform=ax2.transAxes, weight='bold', fontsize=fontsize)

    # Third graph for the frequency - distance relation
    hb = ax3.hexbin(dist,freq,gridsize=25, mincnt = 1,cmap=colormap)
    ax3.set_xlabel('Link path length [km]', fontsize=fontsize)
    ax3.set_ylabel('Frequency [GHz]', fontsize=fontsize)
    ax3.tick_params(axis='x', labelsize=fontsize - 2)
    ax3.tick_params(axis='y', labelsize=fontsize - 2)
    ax3.text(0.97, 0.95, 'C', ha='center', va='center', transform=ax3.transAxes, weight='bold', fontsize=fontsize)

    cb = fig.colorbar(hb, ax=ax3,label='Occurence [-]')
    cb.set_label(label='Occurence [-]', size=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/val_freq_dist.png', dpi=400)
    plt.show()

plot_validationset_freq_dist('copper_r', fontsize=16)


def plot_trainingloss_testloss_allruns(folder_with_runs: str):
    files = [f for f in listdir(folder_with_runs)]
    f, ax1 = plt.subplots(1,1)
    first_plot = True
    for f in files:
        # Load the data
        data = pd.read_csv(folder_with_runs + '/' + f)

        # Check for the first one to have a color
        if first_plot: color = 'black'
        else: color = 'lightgrey'

        ax1.plot(data['loss_trans'],'o-',c=color)

        # Reset the first_plot boolean after the first plotting commmand
        first_plot = False

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss: RDF MSE [mm/h]')
    #ax1.set_title('Validation loss')
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/ex_loss_epochs.png')
    plt.show()


plot_trainingloss_testloss_allruns('C:/Users/ludod/Documents/MSc Thesis/progress_example')



def plot_3panel_seq_len(filename: str):
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    dataset = pd.read_csv(filename, header=0)


filename ='C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/output_LSTM_RDF_test_trans_features.txt'
dataset= pd.read_csv(filename)
f = plt.figure()
ax = f.add_subplot(111)
outputs = dataset['outputs']
targets = dataset['targets']
cv = np.std(outputs) / np.mean(outputs)
r2 = np.square(np.corrcoef(outputs, targets)[0, 1])
RMSE= np.sqrt(np.mean(np.square((outputs - targets))))
Cv_str = 'CV: ' + str(round(cv, 4))
r2_str = '$\mathregular{R^{2}: }$' + str(round(r2, 3))
RMSE_str = 'RMSE: ' + str(round(RMSE,4))

plt.scatter(outputs, targets, c='black', s=4)
plt.title('Predicted vs. observed 15-min rainfall intensities')
plt.xlabel('Model prediction [mm/h]')
plt.ylabel('Observation [mm/h]')
plt.text(0.9, 0.9, Cv_str, ha='center', va='center', transform=ax.transAxes)
plt.text(0.9, 0.95, r2_str, ha='center', va='center', transform=ax.transAxes)
plt.text(0.9,0.85,RMSE_str,ha='center',va='center',transform=ax.transAxes)
plt.show()