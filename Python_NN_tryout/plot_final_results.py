# Script with a few functions that visualize the data that I have
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from matplotlib.dates import DateFormatter
from datetime import datetime
def plt_model_observed_CV_r2(filename: str, fontsize: int):
    dataset = pd.read_csv(filename)
    f = plt.figure()
    ax = f.add_subplot(111)

    combi_str_1, outputs,targets = stat_string_from_dataset(dataset)
    cmap = plt.get_cmap('copper_r',10)
    h1 = ax.hexbin(outputs, targets, cmap=cmap,bins='log',gridsize=50, mincnt=1)
    #plt.scatter(outputs,targets,s=4)
    plt.xlabel('Predicted precipitation rate [mm/h]', fontsize=fontsize)
    plt.ylabel('Observed precipitation rate [mm/h]', fontsize=fontsize)
    #plt.xlim([0.41361,0.41363])
    ax.tick_params(axis='x',labelsize = fontsize-6)
    ax.xaxis.set_label_coords(0.5,-0.09)
    ax.tick_params(axis='y', labelsize= fontsize - 6)
    plt.ticklabel_format(axis='x', style='sci')
    ax.text(0.95, 0.9, combi_str_1, ha='right', va='center', transform=ax.transAxes)
    cb = f.colorbar(h1,ax=ax, label='Occurence [-]')
    cb.set_label(label='Occurence [-]', size=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/scatter_final.png')
    plt.show()

filename = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_8_5_fin_1CPU_5runs_smallgood.txt'
plt_model_observed_CV_r2(filename, fontsize=14)

def plt_onlyrain_model_observed_CV_r2(filename: str, fontsize: int):
    dataset = pd.read_csv(filename)
    f = plt.figure()
    ax = f.add_subplot(111)
    outputs = dataset['outputs_nontrans']
    targets = dataset['targets_nontrans']
    # Calculate the statistics for the plot and add these to the plot
    cv = np.std(outputs) / np.mean(outputs)
    r2 = np.square(np.corrcoef(outputs, targets)[0, 1])
    RMSE= np.sqrt(np.mean(np.square((outputs - targets))))
    Cv_str = 'CV: ' + str(round(cv, 4))
    r2_str = '$\mathregular{R^{2}: }$' + str(round(r2, 3))
    RMSE_str = 'RMSE: ' + str(round(RMSE,4))
    cmap = plt.get_cmap('copper_r', 10)
    plt.hexbin(outputs, targets, cmap=cmap,bins='log',gridsize=50, mincnt=1)
    #plt.scatter(outputs,targets,s=4)
    #plt.title('Predicted vs. observed 15-min rainfall intensities')
    plt.xlabel('Predicted precipitation rate [mm/h]', fontsize=fontsize)
    plt.ylabel('Observed precipitation rate [mm/h]', fontsize=fontsize)
    plt.tick_params(axis='x',labelsize = fontsize-2)
    plt.tick_params(axis='y', labelsize= fontsize - 2)

    plt.text(0.85, 0.9, Cv_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.85, 0.95, r2_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.85,0.85,RMSE_str,ha='center',va='center',transform=ax.transAxes)
    cb = plt.colorbar(label='Occurence [-]')
    cb.set_label(label='Occurence [-]', size=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/scatter_onlyRain_final.png')
    plt.show()

plt_onlyrain_model_observed_CV_r2('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Python_NN_tryout/output_LSTM_UTC_numl_2_hids_128_nepo_10_lr00005_10link_nontransformed_onlyrain.txt', fontsize=14)
filename = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_8_5_fin_1CPU_5runs_smallgood.txt'
plt_model_observed_CV_r2(filename, fontsize=14)



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


def plot_trainingloss_testloss_allruns(folder_with_runs: str,fontsize:int):
    files = [f for f in listdir(folder_with_runs)]
    fig = plt.figure(figsize=(11,11))
    ax1 = fig.add_subplot(111)
    first_plot = True
    number_failed = 0
    for f in files:
        # Load the data
        data = pd.read_csv(folder_with_runs + '/' + f)
        if len(data) <10:
            number_failed +=1
            continue
        # Check for the first one to have a color
        if first_plot:
            color = 'black'
            zorder = 2
        else:
            color = 'lightgrey'
            zorder = 1

        ax1.plot(data['loss_trans'],'o-',c=color,ms=10, zorder=zorder)

        # Reset the first_plot boolean after the first plotting commmand
        first_plot = False
    print(number_failed)
    ax1.set_xlabel('Epochs',fontsize=fontsize)
    ax1.set_ylabel('Loss: RDF MSE',fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize - 2)
    ax1.tick_params(axis='y', labelsize=fontsize - 2)
    ax1.set_ylim([0.3,0.8])
    #ax1.set_title('Validation loss')
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/ex_loss_epochs.png')
    plt.show()


plot_trainingloss_testloss_allruns('C:/Users/ludod/Documents/MSc Thesis/epoch_results_10links_good',fontsize=20)



def plot_9panel_seq_len(filelist: list, raylist: list, fontsize: int):

    outputdict = {}
    targetdict = {}
    combistrdict = {}

    f = plt.figure(figsize=(35,25))
    cmap = plt.get_cmap('copper_r', 10)
    ax1 = f.add_subplot(331)
    ax2 = f.add_subplot(332)
    ax3 = f.add_subplot(333)
    ax4 = f.add_subplot(334)
    ax5 = f.add_subplot(335)
    ax6 = f.add_subplot(336)
    ax7 = f.add_subplot(337)
    ax8 = f.add_subplot(338)
    ax9 = f.add_subplot(339)

    for i in range(len(filelist)):
        j = i+1
        keyname = 'dataset'+str(j)
        dataset = pd.read_csv(filelist[i])
        combistrdict[keyname],outputdict[keyname],targetdict[keyname] = stat_string_from_dataset(dataset, from_ray = raylist[i])

    h1 = ax1.hexbin(outputdict['dataset1'], targetdict['dataset1'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h2 = ax2.hexbin(outputdict['dataset2'], targetdict['dataset2'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h3 = ax3.hexbin(outputdict['dataset3'], targetdict['dataset3'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h4 = ax4.hexbin(outputdict['dataset4'], targetdict['dataset4'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h5 = ax5.hexbin(outputdict['dataset5'], targetdict['dataset5'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h6 = ax6.hexbin(outputdict['dataset6'], targetdict['dataset6'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h7 = ax7.hexbin(outputdict['dataset7'], targetdict['dataset7'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h8 = ax8.hexbin(outputdict['dataset8'], targetdict['dataset8'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)
    h9 = ax9.hexbin(outputdict['dataset9'], targetdict['dataset9'], cmap=cmap,bins='log', gridsize=40, mincnt=1)#, vmin=1, vmax=120000)

    # Set the axis ticks and labels
    for axis in [ax2, ax3,ax5,ax6]:
        axis.axes.xaxis.set_visible(False)
        axis.axes.yaxis.set_visible(False)
    for axis in [ax7,ax8,ax9]:
        axis.tick_params(axis='x',labelsize=fontsize)
        if axis in [ax8,ax9]:  axis.axes.yaxis.set_visible(False)
    for axis in [ax1, ax4, ax7]:
        axis.tick_params(axis='y', labelsize=fontsize)
        if axis in [ax1,ax4]: axis.axes.xaxis.set_visible(False)

    for axes in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
        axes.set_xlim([-1.0,5.0])

    ax8.set_xlabel('Predicted precipitation rate [mm/h]', fontsize=fontsize+4)
    ax4.set_ylabel('Observed precipitation rate [mm/h]', fontsize = fontsize+4)

    # Set the statistics text in the figure
    ax1.text(0.95, 0.9, combistrdict['dataset1'],ha='right', va='center', transform=ax1.transAxes,fontsize=fontsize-2)
    ax1.text(0.05, 0.95, 'A: SL=4', ha='left', va='center', weight = 'bold',transform=ax1.transAxes,fontsize=fontsize)
    ax2.text(0.95, 0.9, combistrdict['dataset2'], ha='right', va='center', transform=ax2.transAxes,fontsize=fontsize-2)
    ax2.text(0.05, 0.95, 'B: SL=8', ha='left', va='center', weight = 'bold',transform=ax2.transAxes,fontsize=fontsize)
    ax3.text(0.95, 0.9, combistrdict['dataset3'], ha='right', va='center', transform=ax3.transAxes,fontsize=fontsize-2)
    ax3.text(0.05, 0.95, 'C: SL=12', ha='left', va='center', weight = 'bold',transform=ax3.transAxes,fontsize=fontsize)
    ax4.text(0.95, 0.9, combistrdict['dataset4'], ha='right', va='center', transform=ax4.transAxes,fontsize=fontsize-2)
    ax4.text(0.05, 0.95, 'D: BS=32', ha='left', va='center',weight = 'bold', transform=ax4.transAxes,fontsize=fontsize)
    ax5.text(0.95, 0.9, combistrdict['dataset5'], ha='right', va='center', transform=ax5.transAxes,fontsize=fontsize-2)
    ax5.text(0.05, 0.95, 'E: BS=64', ha='left', va='center', weight = 'bold',transform=ax5.transAxes,fontsize=fontsize)
    ax6.text(0.95, 0.9, combistrdict['dataset6'], ha='right', va='center', transform=ax6.transAxes,fontsize=fontsize-2)
    ax6.text(0.05, 0.95, 'F: BS=128', ha='left', va='center',weight = 'bold', transform=ax6.transAxes,fontsize=fontsize)
    ax7.text(0.95, 0.9, combistrdict['dataset7'], ha='right', va='center', transform=ax7.transAxes,fontsize=fontsize-2)
    ax7.text(0.05, 0.95, 'G: LF=RDF MSE', ha='left', va='center',weight = 'bold', transform=ax7.transAxes,fontsize=fontsize)
    ax8.text(0.95, 0.9, combistrdict['dataset8'], ha='right', va='center', transform=ax8.transAxes,fontsize=fontsize-2)
    ax8.text(0.05, 0.95, 'H: LF=MSE', ha='left', va='center', weight = 'bold',transform=ax8.transAxes,fontsize=fontsize)
    ax9.text(0.95, 0.9, combistrdict['dataset9'], ha='right', va='center', transform=ax9.transAxes,fontsize=fontsize-2)
    ax9.text(0.05, 0.95, 'I: LF=RDF MAE', ha='left', va='center', weight = 'bold',transform=ax9.transAxes,fontsize=fontsize)


    cb = plt.colorbar(h3, ax=[ax4,ax5,ax6],label='Occurence [-]')
    cb.set_label(label='Occurence [-]', size=fontsize)
    cb.ax.tick_params(labelsize=fontsize)

    ax2.patch.set_facecolor('azure')
    ax4.patch.set_facecolor('azure')
    ax7.patch.set_facecolor('azure')
    for pos in ['top','left','bottom', 'right']:
        ax2.spines[pos].set_linewidth(2)
        ax4.spines[pos].set_linewidth(2)
        ax7.spines[pos].set_linewidth(2)

    f.subplots_adjust(right=0.75,wspace=0.1,hspace=0.1)
    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/ex_seq_len.png')
    plt.show()

filename ='C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_DUMMY_NOTBEST.txt'
filename_default = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_8_5_fin_1CPU_5runs_smallgood.txt'
filename_BS64 = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_fin_2_BS64_1run.txt'
filename_BS128 = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_fin_2_BS128_1run.txt'
filename_LFMAE = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_fin_LFMAE_1run.txt'
filename_LFMSE = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_fin_LFMSE_1run.txt'
filename_SL4 = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_fin_S2_L4_1run.txt'
from_ray_list = [False,True,True,
                 True,False,False,
                 True,False,False]
plot_9panel_seq_len([filename_SL4,filename_default,filename,
                     filename_default,filename_BS64,filename_BS128,
                     filename_default,filename_LFMSE,filename_LFMAE], from_ray_list,fontsize=20)



def stat_string_from_dataset(dataset, from_ray = True):
    # Create the x and y data for the different files
    if not from_ray:
        outputs1 = dataset['outputs']
        targets1 = dataset['targets']
    else:
        outputs1 = dataset['outputs_real']
        targets1 = dataset['targets_real']

    # Calculate the statistics for the plot and add these to the plot
    cv_1 = np.std(outputs1) / np.mean(outputs1)
    r2_1 = np.square(np.corrcoef(outputs1, targets1)[0, 1])
    RMSE_1 = np.sqrt(np.mean(np.square((outputs1 - targets1))))

    combi_str_1 = '$\mathregular{R^{2}: }$' + str(round(r2_1, 3)) + '\n'+\
                  'CV: ' + str(round(cv_1, 4))+ '\n' + \
                  'RMSE: ' + str(round(RMSE_1, 3))

    return combi_str_1, outputs1, targets1


def CML_signal(filename, default_filename,faulty_filename):
    headers = ['ID','SITE_LON','SITE_LAT','FAR_LON','FAR_LAT','FREQ','DATE','RXMIN','RXMAX',
               'DIST','x_site','y_site','x_far','y_far','PROV','AVG_LAT','AVG_LON','TARG_PRCP']
    data = pd.read_csv(filename, header=0, names=headers)
    default_data = pd.read_csv(default_filename)
    faulty_data = pd.read_csv(faulty_filename)
    data['DATETIME'] = pd.to_datetime(data['DATE'],format='%Y%m%d%H%M')
    minsign = data['RXMIN']
    maxsign = data['RXMAX']

    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)

    # First plot
    ax1.bar(data['DATETIME'],data['TARG_PRCP'],color='lightblue',width=0.01,label='Observed prec.')
    ax1.step(data['DATETIME'],default_data['outputs_real'][0:88],color='darkblue', label='Predicted prec.')
    axes2 = plt.twinx(ax=ax1)
    axes2.plot(data['DATETIME'],maxsign,label='Max. received signal',color='darkred')
    axes2.plot(data['DATETIME'],minsign, label='Min. received signal',color='darkorange')
    myFmt = DateFormatter('%d-%m %H:00')
    #axes2.set_ylabel('Received signal level [mdB]')
    #ax1.set_ylabel('Precipitation rate [mm/h]')
    axes2.set_ylim([-68,-44])
    ax1.xaxis.set_major_formatter(myFmt)

    # Second plot
    ax2.bar(data['DATETIME'], data['TARG_PRCP'], color='lightblue', width=0.01, label='Observed prec.')
    ax2.step(data['DATETIME'], faulty_data['outputs_real'][0:88], color='darkblue', label='Predicted prec.')
    axes22 = plt.twinx(ax=ax2)
    axes22.plot(data['DATETIME'], maxsign, label='Max. received signal', color='darkred')
    axes22.plot(data['DATETIME'], minsign, label='Min. received signal', color='darkorange')

    axes22.set_ylabel('Received signal level [mdB]')
    axes22.yaxis.set_label_coords(1.1,1.15)
    ax2.set_ylabel('Precipitation rate [mm/h]')
    ax2.yaxis.set_label_coords(-0.05,1.15)
    axes22.set_ylim([-68, -44])
    ax2.xaxis.set_major_formatter(myFmt)
    ax1.text(0.03, 0.92, 'A', ha='left', va='center', weight='bold', transform=ax1.transAxes)
    ax2.text(0.03, 0.92, 'B', ha='left', va='center', weight='bold', transform=ax2.transAxes)
    f.autofmt_xdate()
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1,axes2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines,labels,loc='upper center',ncol=2, fontsize=9)

    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/CML_timesignal_faulty')
    plt.show()

filename_default = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_8_5_fin_1CPU_5runs_smallgood.txt'
filename_faulty = 'C:/Users/ludod/Documents/MSc Thesis/best_result_10_links/output_DUMMY_NOTBEST.txt'
CML_signal('C:/Users/ludod/Documents/MSc Thesis/rain_sample_test.txt', filename_default, filename_faulty)


def CML_signal_raw(filename):
    headers = ['ID','SITE_LON','SITE_LAT','FAR_LON','FAR_LAT','FREQ','DATE','RXMIN','RXMAX',
               'DIST','x_site','y_site','x_far','y_far','PROV','AVG_LAT','AVG_LON','TARG_PRCP']
    data = pd.read_csv(filename, header=0, names=headers)
    data['DATETIME'] = pd.to_datetime(data['DATE'],format='%Y%m%d%H%M')
    minsign = data['RXMIN']
    maxsign = data['RXMAX']

    f = plt.figure()
    ax1 = f.add_subplot()

    # First plot
    ax1.bar(data['DATETIME'],data['TARG_PRCP'],color='lightblue',width=0.01,label='Observed prec.')
    axes2 = plt.twinx(ax=ax1)
    axes2.plot(data['DATETIME'],maxsign,label='Max. received signal',color='darkred')
    axes2.plot(data['DATETIME'],minsign, label='Min. received signal',color='darkorange')
    myFmt = DateFormatter('%d-%m %H:00')
    axes2.set_ylim([-68,-44])
    ax1.xaxis.set_major_formatter(myFmt)

    axes2.set_ylabel('Received signal level [mdB]')
    #axes2.yaxis.set_label_coords(1.1, 1.15)
    ax1.set_ylabel('Precipitation rate [mm/h]')
    #ax1.yaxis.set_label_coords(-0.05, 1.15)

    f.autofmt_xdate()
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, axes2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines,labels,loc='upper center',ncol=3, fontsize=9)

    plt.savefig('C:/Users/ludod/Documents/GitHub/Thesis_HWM2021/Thesis/images/CML_timesignal_raw')
    plt.show()

CML_signal_raw('C:/Users/ludod/Documents/MSc Thesis/rain_sample_test.txt')




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