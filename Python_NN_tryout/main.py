# This is a sample Python script.

# First example python script. The script allows to switch between the locally stored complete dataset (not publicly available)
# and a sample dataset of one day that can be used to try out the script.
# TODO: implement switch between local dataset on my PC and a smaller example dataset

# Import the necessary functions and modules
import data_functions as df
import numpy as np
import matplotlib.pyplot as plt

# Load the CML data and match the right rainfall amount to the links
data = df.load_CML_data('20110714', 'NOKIA')
data_with_target, match_set, lon_grid, lat_grid = df.ReadRainLocation(data)

# For now, remove the erroneous radar data from the dataset. Will be fixed later
data_with_target_errorless = data_with_target[data_with_target['TARG_PRCP'] != 65535.00]

# Filter out the erroneous CML data values (-1.0)
data_with_target_errorless = data_with_target_errorless[data_with_target_errorless['RXMIN'] != -1.0]
data_with_target_errorless = data_with_target_errorless[data_with_target_errorless['RXMAX'] != -1.0]

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

#####################################
############### RNN #################
#####################################

import torch
import torch.nn as nn
import torchvision
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter(filename_suffix='20 epochs lr 0.00003 batch 64')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters to be defined
num_classes = 1
num_epochs = 20
batch_size = 64 # originally 64
learning_rate = 0.00003

input_size = 2
sequence_length = 4 # originally 64
hidden_size = 128 # originally 128
num_layers = 4

# 100 bins from 0 to 5 to categorize the target variable
bins = np.arange(0,4.95,0.05)

# Prepare the input data to the RNN model
# To fit the batches in the dataset, use data up until the batch size fits in the total length

def load_data(batch_size):
    data_len_train = int(np.floor(len(train)/batch_size) * batch_size)
    data_len_test = int(np.floor(len(test)/batch_size) * batch_size)
    data_len_val = int(np.floor(len(val) / batch_size) * batch_size)

    features_train = torch.tensor(train[['RXMIN','RXMAX']].iloc[0:data_len_train].values).type(torch.DoubleTensor)
    features_test = torch.tensor(test[['RXMIN','RXMAX']].iloc[0:data_len_train].values).type(torch.DoubleTensor)
    features_val = torch.tensor(val[['RXMIN','RXMAX']].iloc[0:data_len_val].values).type(torch.DoubleTensor)

    targets_train = torch.from_numpy(train['TARG_PRCP'].iloc[0:data_len_train].values).type(torch.DoubleTensor)
    targets_test = torch.from_numpy(test['TARG_PRCP'].iloc[0:data_len_test].values).type(torch.DoubleTensor)
    targets_val = torch.from_numpy(val['TARG_PRCP'].iloc[0:data_len_val].values).type(torch.DoubleTensor)

    # Transform the target data using a log transform. Add 1 to avoid mathematical errors when taking the logarithm of zero
    targets_train_transform = np.log10(targets_train+1)
    targets_test_transform  = np.log10(targets_test+1)
    targets_val_transform = np.log10(targets_val + 1)

    # Normalize the target data w.r.t. the full range of the data
    targets_test_norm = (targets_test_transform - targets_test_transform.min()) / (targets_test_transform.max() - targets_test_transform.min())
    targets_train_norm = (targets_train_transform - targets_train_transform.min()) / (targets_train_transform.max() - targets_train_transform.min())
    targets_val_norm = (targets_val_transform - targets_val_transform.min()) / (targets_val_transform.max() - targets_val_transform.min())

    # targets_test = torch.tensor(test_sorted['TARG_PRCP'].values).type(torch.LongTensor)
    # targets_test = torch.histc(targets_test,bins=100,min=0,max=5)

    features_train_norm = (features_train - features_train.min()) / (features_train.max() - features_train.min())
    features_test_norm = (features_test - features_test.min()) / (features_test.max() - features_test.min())
    features_val_norm = (features_val - features_val.min()) / (features_val.max() - features_val.min())

    train_pytorch = TensorDataset(features_train_norm,targets_train_norm)
    test_pytorch  = TensorDataset(features_test_norm,targets_test_norm)
    val_pytorch = TensorDataset(features_val_norm, targets_val_norm)

    return train_pytorch, test_pytorch, val_pytorch


# train_pytorch, test_pytorch = load_data()
# train_loader = DataLoader(train_pytorch, batch_size = batch_size, shuffle=False)
# test_loader = DataLoader(test_pytorch, batch_size = batch_size,shuffle=False)

# Create the RNN model
class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNNModel,self).__init__()

        # Number of hidden dimensions
        self.hidden_size = hidden_size

        # Number of hidden layers
        self.num_layers = num_layers

        # RNN
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first = True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,x):

        # Initialize hidden state with zeros
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size)).double()

        # One time step
        out, hn = self.rnn(x,h0)
        # many to one RNN: get the last result
        out = self.fc(out[:,-1,:])

        return out

def train_cifar(config, checkpoint_dir=None, data_dir=None):
    model = RNNModel(input_size, config["hidden_size"],config["num_layers"],num_classes)
    model = model.double() # Set the model to double so it does not expect floats

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() >1:
            model = nn.DataParallel(model)
    model.to(device)

    error = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset, valset = load_data(batch_size)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    count = 0
    MSE = 0
    for epoch in range(config['num_epochs']):
        for i, (images, labels) in enumerate(train_loader):

            # train_set = torch.autograd.Variable(images.view(-1, sequence_length, input_size)).double()
            train_set = images.reshape(-1, sequence_length, input_size)
            # train = torch.autograd.Variable(images.view(-1, len(images)))
            labels = torch.autograd.Variable(labels)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train_set)

            # Calculate softmax and MSE Loss
            loss = error(outputs, labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            count += 1

            if count % 400 == 0:
                # Calculate Accuracy
                print('count is: ', count)
                correct = 0
                MSE_per_batch = []
                total = 0

                # Keep track of where you are in the test loop
                test_count = 0
                # Iterate through test dataset
                for i, (images, labels) in enumerate(test_loader):
                    # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
                    test_set = images.reshape(-1, sequence_length, input_size)
                    test_set, labels = test_set.to(device), labels.to(device)

                    # Forward propagation
                    outputs = model(test_set)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]  # Useful when you're looking at a probability distribution

                    # Total number of labels
                    total += labels.size(0)

                    MSE_per_batch.append((outputs - labels).square().mean().item())
                    # accuracy = 100 * correct / float(total)
                MSE = np.average(MSE_per_batch)

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                MSE_list.append(MSE)
                #writer.add_scalar('Test loss', MSE, count)
                if count % 800 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  MSE: {} %'.format(count, loss.data.item(), MSE))

                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=MSE)
    print('Finished training')
    #writer.close()

def val_accuracy(config, model, device='cpu'):
    _,_,valset = load_data(batch_size)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, labels in val_loader:
            # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
            val_set = images.reshape(-1, sequence_length, input_size)
            val_set, labels = val_set.to(device), labels.to(device)
            # Forward propagation
            outputs = model(val_set)

            MSE_per_batch.append((outputs - labels).square().mean().item())
            # accuracy = 100 * correct / float(total)
        MSE = np.average(MSE_per_batch)
    return MSE

def main(num_samples = 4, gpus_per_trial=0):

    config = {
        #'batchsize': tune.choice([16, 32, 64, 128, 256]),
        'lr': tune.loguniform(1e-6, 1e-3),
        'num_epochs': tune.randint(4, 8),
        'hidden_size': tune.choice([64,128,256]),#tune.choice([16, 32, 64, 128, 256]),
        'num_layers': tune.choice([2,4,6]),#tune.choice([2, 4, 8])
    }
    load_data(batch_size)
    scheduler = ASHAScheduler(metric='loss',mode='min',grace_period=1,reduction_factor=2)
    reporter = CLIReporter(metric_columns='loss')
    result = tune.run(train_cifar,
                      resources_per_trial={'cpu': 4, 'gpu': gpus_per_trial},
                      config=config,
                      num_samples=num_samples,
                      scheduler=scheduler,
                      progress_reporter=reporter)

                      #checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))


    best_trained_model = RNNModel(input_size, best_trial.config["hidden_size"], best_trial.config["num_layers"],num_classes)
    best_trained_model.double()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here and the number of samples taken:
    main(num_samples=5)




#################
# OLD MODEL RUN # This model run works, beforehand is the optimizing tuning algorithm implemented
#################

model = RNNModel(input_size,hidden_size,num_layers,num_classes)

# Cross Entropy Loss
error = nn.MSELoss() # Use MSELoss instead of CrossEntropyLoss, since CEL needs labels and classification

# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
iteration_list = []
MSE_list = []
count = 0
model = model.double()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # train_set = torch.autograd.Variable(images.view(-1, sequence_length, input_size)).double()
        train_set = images.reshape(-1, sequence_length, input_size)
        # train = torch.autograd.Variable(images.view(-1, len(images)))
        labels = torch.autograd.Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train_set)

        # Calculate softmax and MSE Loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        if count % 400 == 0:
            # Calculate Accuracy
            print('count is: ', count)
            correct = 0
            MSE_per_batch = []
            total = 0

            # Keep track of where you are in the test loop
            test_count = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
                test_set = images.reshape(-1,sequence_length, input_size)

                # Forward propagation
                outputs = model(test_set)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]  # Useful when you're looking at a probability distribution

                # Total number of labels
                total += labels.size(0)

                MSE_per_batch.append((outputs - labels).square().mean().item())
                # accuracy = 100 * correct / float(total)
            MSE = np.average(MSE_per_batch)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            MSE_list.append(MSE)
            writer.add_scalar('Test loss', MSE, count)
            if count % 800 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  MSE: {} %'.format(count, loss.data.item(), MSE))

writer.close()

#plt.imshow(target_data_nc_prcp, vmin=0, vmax=0.05)
#plt.colorbar()
#plt.show()


