# Script with a few functions that visualize the data that I have
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plt_model_observed_CV_r2(filename: str):
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

    plt.scatter(outputs, targets, c='black', s=4)
    plt.title('Predicted vs. observed 15-min rainfall intensities')
    plt.xlabel('Model prediction [mm/h]')
    plt.ylabel('Observation [mm/h]')
    plt.text(0.9, 0.9, Cv_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.9, 0.95, r2_str, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.9,0.85,RMSE_str,ha='center',va='center',transform=ax.transAxes)
    plt.show()

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