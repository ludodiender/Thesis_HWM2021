# This is a sample Python script.

# First example python script. The script allows to switch between the locally stored complete dataset (not publicly available)
# and a sample dataset of one day that can be used to try out the script.
# TODO: implement switch between local dataset on my PC and a smaller example dataset
'''
# Import the necessary functions and modules
import importlib
import data_functions as df
import dataclass_functions as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

importlib.reload(df)
importlib.reload(dc)

# Load the CML data and match the right rainfall amount to the links
data17 = df.load_CML_data('NOKIA',date_selected='20110717')
data = df.load_CML_data('NOKIA',date_selected='20110716')

data_ID = data[:]
matching_ID_table, lon_grid, lat_grid = df.create_ID_table(data_ID)
data, matching_ID_table = df.generate_ID(data, matching_ID_table, lon_grid, lat_grid)
data_with_target = df.ReadRainLocation(data,matching_ID_table)
data_with_target_errorless = df.filter_data_error(data_with_target)
df.write_to_csv(data_with_target_errorless, 'NOKIA')

files = df.data_to_harddrive('D:/CML_data_NL/NOKIA_data/NOKIA_data',provider='NOKIA')
df.data_to_harddrive('D:/CML_data_NL/NOKIA_data/NOKIA_data', provider='NOKIA',longrid=lon_grid,latgrid=lat_grid,match_table=matching_ID_table_2011_2012_unique)

# Split the data in a training, test and validation set
train,test,val = df.split_data(data_with_target_errorless,40,40,20)

######### INSPECT THE LOADED DATA ###############
# Load the radar data separately to inspect
target_data_nc_prcp,target_data_nc = df.load_radar_data_netcdf(20110724,1145)

# Plot target radar data for a specific timestep to check
plt.imshow(target_data_nc_prcp.data , extent=[np.min(lon_grid),np.max(lon_grid),np.min(lat_grid),np.max(lat_grid)], vmin=0, vmax=0.31)
plt.grid()
plt.show()

# Plot the target radar data without errors to inspect any spatial structures in the error pattern
plt.scatter(data_with_target_errorless['AVG_LON'],data_with_target_errorless['AVG_LAT'],c=data_with_target_errorless['TARG_PRCP'])
plt.show()

# Inspect the relation between received signal level and the target radar precipitation.
plt.scatter(data_with_target_errorless['RXMAX'], data_with_target_errorless['TARG_PRCP'])
plt.show()

plt.hist(train['ID'].value_counts(),bins=100, range=[30,50])
plt.show()
'''

#####################################
############### RNN #################
#####################################

import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import time
from datetime import datetime
import dataclass_functions as dc
from functools import partial
import time
import importlib
from pathlib import Path

importlib.reload(dc)

# Initiliaze the writer and the device
# writer = SummaryWriter(filename_suffix='20 epochs lr 0.00003 batch 64')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_file_name = 'output_cpu_small_seql_8_2ep_updatedRDF_highlr.txt'

# Hyper parameter setting, only needed when not tuning using RayTune
num_classes = 1
num_epochs = 2
batch_size = 256
learning_rate = 0.001
input_size = 2
sequence_length = 8
hidden_size = 128
num_layers = 4

# Set starting time
t_start = time.time()

# Define the header names for the data
header_names = ['ID', 'SITE_LAT', 'SITE_LON', 'FAR_LAT', 'FAR_LON', 'FREQ', 'DATE', 'RXMIN', 'RXMAX', 'DIST',
                'P4_start.v', 'P4_start.u', 'P4_end.v', 'P4_end.u', 'PROV', 'AVG_LON', 'AVG_LAT', 'TARG_PRCP']


# Set up the dataset class for the CML dataset

class MyCMLDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, seq_len: int, headers: list):
        self.directory = Path(data_dir)
        self.files = sorted((f, dc.get_sample_count_by_file(f)) for f in self.directory.iterdir() if f.is_file())
        self.total_sample_count = sum(f[-1] for f in self.files)
        self.header_names = headers

        self.seq_len = seq_len

        self.run_sample_count = 0
        self.current_count = 0
        self.index_in_list = -1
        self.data = []
        self.dates = []

        self.current_iter = 0
        self.cache_iter = 0
        self.invocation_count = 0

    def __getitem__(self, item):
        self.invocation_count += 1

        if self.current_iter != self.cache_iter:
            self.index_in_list = -1
            self.current_count = 0
            self.run_sample_count = 0
            # print('Current iter:',self.current_iter,', Cache iter:',self.cache_iter,', Epoch passed.')

        if not self.current_count <= item < self.run_sample_count:
            self.index_in_list += 1
            file_, sample_count_file = self.files[self.index_in_list]
            self.current_count = self.run_sample_count
            self.run_sample_count = self.run_sample_count + sample_count_file
            # print(self.run_sample_count,'file sample count:',sample_count_file)

            with file_.open() as f:
                data_raw = pd.read_csv(f, sep=',', header=0, names=self.header_names)
                process_data, sep_dates = dc.load_data_separate(data_raw)
                self.data = process_data
                self.dates = sep_dates
                print(self.index_in_list, 'loaded, datasize:', len(self.data))

        # now file_ has sample_count samples
        file_idx = item - self.current_count  # the index we want to access in file_
        self.current_iter = self.cache_iter

        if file_idx < (self.seq_len - 1):
            return None
        elif not dc.DatesCheck(self.dates[file_idx - (self.seq_len - 1):file_idx + 1]):
            return None
        else:
            return self.data.tensors[0][file_idx - (self.seq_len - 1):file_idx + 1], self.data.tensors[1][file_idx]

    def __len__(self):
        return self.total_sample_count - self.seq_len


# Set up the RNN model to be used
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_size = hidden_size

        # Set device
        self.device = device

        # Number of hidden layers
        self.num_layers = num_layers

        # RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).double().to(device)

        # One time step
        out, hn = self.rnn(x, h0)
        # many to one RNN: get the last result
        out = self.fc(out[:, -1, :])

        return out


# Create the training loop function

def train_cifar(config, output, checkpoint_dir=None, data_dir=None):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    model = RNNModel(input_size, config["hidden_size"], config["num_layers"], num_classes, device)
    model = model.double()  # Set the model to double so it does not expect floats
    print(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    #error = nn.MSELoss()
    error = dc.custom_MSE_RDF_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    #(model.default_resource_request(config=config)._bundles)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    print('Checkpoint:',checkpoint_dir)
    # Load the data by selecting the right dataset
    #trainpath = '/lustre/scratch/WUR/ESG/diend004/CML_RAD_perID/2011_training/NOKIA'
    #testpath = '/lustre/scratch/WUR/ESG/diend004/CML_RAD_perID/2012_testing/NOKIA'

    trainpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_train_testing'
    testpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_test_testing'

    # trainpath = '/lustre/scratch/WUR/ESG/diend004/Testfiles_JupyterNotebook/train'
    # testpath = '/lustre/scratch/WUR/ESG/diend004/Testfiles_JupyterNotebook/test'

    trainset = MyCMLDataset(data_dir=trainpath, seq_len=sequence_length, headers=header_names)
    testset = MyCMLDataset(data_dir=testpath, seq_len=sequence_length, headers=header_names)

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False, drop_last=True,
                              num_workers=1, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False, drop_last=True,
                             num_workers=1, pin_memory=True)

    loss_list = []
    iteration_list = []
    MSE_list = []
    MSE = 0
    for epoch in range(config['num_epochs']):
        running_loss = []
        t_start_train = time.time()
        t_end_train = t_start_train
        time_count = 0
        for i, (features, targets) in enumerate(train_loader):
            train_set = torch.autograd.Variable(features)
            targets = torch.autograd.Variable(targets)
            train_set, targets = train_set.to(device), targets.to(device)
            t_load_train = time.time()

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train_set).squeeze(1)

            # Calculate softmax and MSE Loss
            loss = error(outputs, targets)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()
            t_load_diff = t_load_train - t_end_train
            t_gradients_train = time.time()
            t_grad_diff = t_gradients_train - t_load_train
            t_end_train = time.time()

            time_count += 1
            # print('Current training loss:',loss.data)

            for param in model.parameters():
                if np.count_nonzero(np.isnan(param.detach().numpy())) != 0:
                    print('Parameters NaN from now on, in batch ', time_count)
                    break

            if time_count % 100 == 0:
                print('Loading data: {}  Updating gradients: {}'.format(t_load_diff, t_grad_diff))

            running_loss.append(loss.data)

        running_loss_item = np.average(running_loss)
        loss_list.append(running_loss_item)

        # Calculate Test error
        MSE_per_batch = []
        MSE_per_batch_nontrans = []

        # Iterate through test dataset
        for i, (features, targets) in enumerate(test_loader):
            test_set = torch.autograd.Variable(features)
            targets = torch.autograd.Variable(targets)
            test_set, targets = test_set.to(device), targets.to(device)

            # Forward propagation
            outputs = model(test_set)
            nans_output = np.count_nonzero(np.isnan(outputs.detach().numpy()))
            nans_target = np.count_nonzero(np.isnan(targets.detach().numpy()))
            if (nans_output + nans_target) > 0:
                print('Found', nans_output, 'NaNs in the output and', nans_target, 'NaNs in the target')

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]  # Useful when you're looking at a probability distribution
            targets_nontrans = np.power(10, targets) - 1
            outputs_nontrans = np.power(10, outputs.detach().numpy().reshape(1, -1)) - 1

            # Calculate MSE per batch
            MSE_per_batch.append((outputs - targets).square().mean().item())
            # if np.isnan((outputs-targets).square().mean().item()):
            #    print('MSE for this batch is NaN!')
            MSE_per_batch_nontrans.append((targets_nontrans - outputs_nontrans).square().mean().item())
            # if np.isnan((targets_nontrans - outputs_nontrans).square().mean().item()):
            #    print('MSE_nontrans for this batch is NaN!')

        MSE = np.nanmean(MSE_per_batch)
        MSE_nontrans = np.nanmean(MSE_per_batch_nontrans)
        # store loss and iteration

        MSE_list.append(MSE)

        # Print Loss
        print('Iteration: {}  Loss: {}  MSE: {} Time spent so far: {} %'.format(epoch, running_loss_item, MSE,
                                                                                time.time() - t_start))

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=MSE,
                    loss_trans=MSE_nontrans)  # removed loss=loss.data.item(), think that is the training value and not the validation one

        # Notify the dataset class that the next epoch will take place
        testset.current_iter += 1
        trainset.current_iter += 1

        # with open(output, 'a') as f:
        # write_str = str(epoch)+','+ str(config['num_epochs']) +','+ str(config['hidden_size'])  +','+ str(config['num_layers']) +','+str(loss.data.item())
        # f.write(write_str)

    print('Finished training')


def val_accuracy(model, output_Name, device='cpu'):
    #valpath = '/lustre/scratch/WUR/ESG/diend004/CML_RAD_perID/2013_validating/NOKIA'
    valpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_validate_testing'
    # valpath = '/lustre/scratch/WUR/ESG/diend004/Testfiles_JupyterNotebook/validate'

    valset = MyCMLDataset(data_dir=valpath, seq_len=sequence_length, headers=header_names)
    with open(output_Name, 'w') as f:
        f.write('outputs,targets \n')
    val_loader = DataLoader(valset, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False, drop_last=True,
                            num_workers=1, pin_memory=True)
    MSE_per_batch = []
    with torch.no_grad():
        for i, (features, targets) in enumerate(val_loader):
            val_set = torch.autograd.Variable(features)
            targets = torch.autograd.Variable(targets)
            val_set, targets = val_set.to(device), targets.to(device)
            # Forward propagation
            outputs = model(val_set)

            # Inverse the scaling done in load_separate (dataclass_functions) to obtain 'real' predicted and observed values.
            # Inverse the scaling done in load_separate (dataclass_functions) to obtain 'real' predicted and observed values.
            targ_max = np.log10(55.959999084472656 + np.sqrt(np.power(55.959999084472656, 2) + 1))  # Max: found in minmax_values_perdataset.txt
            targ_min = np.log10(0.0 + np.sqrt(np.power(0.0, 2) + 1))  # Min: found in minmax_values_perdataset.txt

            # In scaling the targets and outputs, the first step was to take the log sine and subsequently min-max scale it. Reverse min-max scaling first
            outputs_real_nonlog = (targ_max - targ_min) * outputs + targ_min
            targets_real_nonlog = (targ_max - targ_min) * targets + targ_min

            # Invert the logsine
            outputs_real = (-0.5 / np.power(10, outputs_real_nonlog)) + (0.5 * np.power(10, outputs_real_nonlog))
            targets_real = (-0.5 / np.power(10, targets_real_nonlog)) + (0.5 * np.power(10, targets_real_nonlog))

            with open(output_Name,'a') as f:
                for k in range(1,len(outputs)):
                    output_val = outputs_real.data[k].item()
                    target_val = targets_real.data[k].item()
                    output_val_non = outputs.data[k].item()
                    target_val_non = targets.data[k].item()
                    write_str = str(output_val) + ','+str(target_val)+','+str(output_val_non)+','+str(target_val_non)+'\n'
                    f. write(write_str)

            MSE_per_batch.append((outputs_real - targets_real).square().mean().item())
            # accuracy = 100 * correct / float(total)
        mse = np.average(MSE_per_batch)
    return mse


# Define the main function for RayTune
def main(num_samples=4, gpus_per_trial=0):
    ray.init(num_cpus=8,num_gpus=gpus_per_trial) # Specify num_cpus here
    print('Ray initialized')
    #torch.autograd.set_detect_anomaly(True)
    config = {
        'lr': tune.loguniform(1e-6, 1e-3),
        'num_epochs': tune.randint(1, 2),
        'hidden_size': tune.choice([64, 128, 256]),  # tune.choice([16, 32, 64, 128, 256]),
        'num_layers': tune.choice([2, 4, 6, 8, 16]),  # tune.choice([2, 4, 8])
    }
    # output_Name = 'output_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt'
    output_Name = output_file_name
    # with open(output_Name,'w') as f:
    #    f.write('epoch, num_epochs, hidden_size, num_layers,train_loss \n')

    scheduler = ASHAScheduler(metric='loss', mode='min', grace_period=1, reduction_factor=2)
    reporter = CLIReporter(metric_columns=['loss', 'loss_trans'], max_report_frequency=60)
    print('Reporter and Scheduler set')
    result = tune.run(partial(train_cifar, output=output_Name),# checkpoint_dir='C:/Users/diend004/Documents/MSc_Thesis'),
                      resources_per_trial={'cpu': 7, 'gpu': gpus_per_trial},
                      config=config,
                      num_samples=num_samples,
                      scheduler=scheduler,
                      progress_reporter=reporter)

    # checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    best_trained_model = RNNModel(input_size, best_trial.config["hidden_size"], best_trial.config["num_layers"],
                                  num_classes, device)
    best_trained_model.double()

    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)

    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = val_accuracy(best_trained_model, output_Name, device)
    print("Best trial test set loss: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here and the number of samples taken:
    main(num_samples=1, gpus_per_trial=0)


#################
# OLD MODEL RUN # This model run works, beforehand is the optimizing tuning algorithm implemented
#################
writer = SummaryWriter(filename_suffix='scaled with DIST, 10 epochs')

model = RNNModel(input_size,hidden_size,num_layers,num_classes,device)

# Cross Entropy Loss
# error = nn.MSELoss() # Use MSELoss instead of CrossEntropyLoss, since CEL needs labels and classification
error = dc.custom_MSE_RDF_loss
# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
iteration_list = []
MSE_list = []
count = 0
model = model.double()

delta_loss = 4e-10
old_loss = 10
# 10 links
#trainpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_train_testing'
#testpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_test_testing'
#valpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_validate_testing'

# 1 link
trainpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_Small_testset/train'
testpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_Small_testset/test'
valpath = 'C:/Users/ludod/Documents/MSc Thesis/CML_Small_testset/validate'

train_pytorch = MyCMLDataset(data_dir=trainpath, seq_len=4, headers=header_names)
test_pytorch = MyCMLDataset(data_dir=testpath, seq_len=4, headers=header_names)
val_pytorch= MyCMLDataset(data_dir=valpath, seq_len=4, headers=header_names)

train_loader = DataLoader(train_pytorch, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False,drop_last=True)
test_loader = DataLoader(test_pytorch, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False, drop_last=True)
val_loader = DataLoader(val_pytorch, batch_size=batch_size, collate_fn=dc.my_collate, shuffle=False, drop_last=True)

TuningStop = False

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    j=0
    running_loss_train = []
    for i, (features, targets) in enumerate(train_loader):
        train_set = torch.autograd.Variable(features)
        targets = torch.autograd.Variable(targets)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train_set).squeeze(1)

        # Calculate softmax and MSE Loss
        loss = error(outputs, targets)
        running_loss_train.append(loss.item())
        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

    # Calculate Accuracy
    print('count is: ', count)
    correct = 0
    MSE_per_batch = []
    MSE_per_batch_nontrans = []
    total = 0
    with open(output_file_name, 'w') as f:
        f.write('outputs,targets,outputs_nontrans,targets_nontrans\n')
    # Keep track of where you are in the test loop
    test_count = []
    # Iterate through test dataset
    with torch.no_grad():
        for features,targets in test_loader:
            # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
            test_set = torch.autograd.Variable(features)
            targets = torch.autograd.Variable(targets)
    
            # Forward propagation
            outputs = model(test_set)
            
            #if np.count_nonzero(np.isnan(outputs.detach().numpy())) != 0:
                #print('Output is NaN after count',count)
                
            # Inverse the scaling done in load_separate (dataclass_functions) to obtain 'real' predicted and observed values.
            targ_max = np.log10(107.91999053955078+np.sqrt(np.power(107.91999053955078,2)+1)) # Max: found in minmax_values_perdataset.txt
            targ_min = np.log10(0.0 +np.sqrt(np.power(0.0,2)+1)) # Min: found in minmax_values_perdataset.txt
            
            # In scaling the targets and outputs, the first step was to take the log sine and subsequently min-max scale it. Reverse min-max scaling first
            outputs_real_nonlog = (targ_max - targ_min)*outputs + targ_min
            targets_real_nonlog = (targ_max - targ_min)*targets + targ_min
            
            # Invert the logsine
            outputs_real = ( -0.5 / np.power(10,outputs_real_nonlog) ) + ( 0.5 * np.power(10,outputs_real_nonlog) )
            targets_real = ( -0.5 / np.power(10,targets_real_nonlog) ) + ( 0.5 * np.power(10,targets_real_nonlog) )
            
            if TuningStop:
                with open(output_file_name,'a') as f:
                    for k in range(1,len(outputs)):
                        output_val = outputs_real.data[k].item()
                        target_val = targets_real.data[k].item()
                        output_val_non = outputs.data[k].item()
                        target_val_non = targets.data[k].item()
                        write_str = str(output_val) + ','+str(target_val)+','+str(output_val_non)+','+str(target_val_non)+'\n'
                        f. write(write_str)

                    
            MSE_per_batch.append(dc.custom_MSE_RDF_loss(outputs,targets))
            test_count.append(len(outputs))
            #print('RDF MSE:',dc.custom_MSE_RDF_loss(outputs,targets))
            MSE_per_batch_nontrans.append((targets_real - outputs_real).square().mean().item())
            output_to_calc = outputs
            targets_to_calc = targets
            break
    test_pytorch.current_iter +=1

    MSE = np.average(MSE_per_batch, weights=test_count)
    MSE_nontrans = np.average(MSE_per_batch_nontrans, weights=test_count)

    # store loss and iteration
    batch_train_loss = np.mean(running_loss_train)
    loss_list.append(batch_train_loss)
    iteration_list.append(count)
    MSE_list.append(MSE)
    writer.add_scalar('Test loss', MSE, count)
    writer.add_scalar('Training loss',batch_train_loss,count)
    writer.add_scalar('Test loss (untransformed)',MSE_nontrans,count)
    
    if count % 10 == 0:
        # Print Loss
        print('Iteration: {}  Loss: {}  MSE: {} %'.format(count, loss.data.item(), MSE))

    if TuningStop: break
    # Check if tuning can continue
    if (old_loss - MSE) < delta_loss:
        print('Test loss not decreasing enough anymore, this run is terminated after', epoch, 'epochs.')
        TuningStop = True
    if epoch == (num_epochs - 2):
        TuningStop = True
        print('End of epochs reached.')

    old_loss = MSE

    train_pytorch.current_iter += 1
writer.close()





#plt.imshow(target_data_nc_prcp, vmin=0, vmax=0.05)
#plt.colorbar()
#plt.show()

'''
input = np.arange(1,21).reshape(-1,2)
input = torch.tensor(input,dtype=torch.float)
target_input = np.arange(1,11).reshape(-1,1)
target_input = torch.tensor(target_input, dtype=torch.float)
total_tensor = TensorDataset(input,target_input)
ds = MyDaatset(total_tensor, seq_len=3)
dl = DataLoader(ds,batch_size=5, drop_last=True,collate_fn=my_collate)
train_ds = MyDaatset(train_pytorch,seq_len=4)
train_dl = DataLoader(train_ds,batch_size=64,drop_last=True,collate_fn=my_collate)

i = 0
for inp,label in train_dl:
    if i < 5:
        print(inp.numpy())
        print(label)
        i+=1
    else: break
    
'''

def RDF(y):
    ys = 0.95
    yr = 5
    RDF = torch.where(y < 0, 3.00,(1-ys*np.exp(-yr * y)) )

    return RDF