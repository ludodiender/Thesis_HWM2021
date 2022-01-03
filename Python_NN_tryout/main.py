# This is a sample Python script.

# First example python script. The script allows to switch between the locally stored complete dataset (not publicly available)
# and a sample dataset of one day that can be used to try out the script.
# TODO: implement switch between local dataset on my PC and a smaller example dataset

# Import the necessary functions and modules

import data_functions as df
import importlib
importlib.reload(df)
import numpy as np
import matplotlib.pyplot as plt

# Load the CML data and match the right rainfall amount to the links
data,matching_ID_table = df.load_CML_data('NOKIA',date_selected='20110716')
data_with_target, match_set, lon_grid, lat_grid = df.ReadRainLocation(data,matching_ID_table)
data_with_target_errorless = df.filter_data_error(data_with_target)
df.write_to_csv(data_with_target_errorless, 'NOKIA')

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
writer = SummaryWriter(filename_suffix='20 epochs lr 0.00003 batch 64')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters to be defined
num_classes = 1
num_epochs = 20
batch_size = 64 # originally 64
learning_rate = 0.00008

input_size = 2
sequence_length = 4 # originally 64
hidden_size = 128 # originally 128
num_layers = 4

# 100 bins from 0 to 5 to categorize the target variable
bins = np.arange(0,4.95,0.05)

# Prepare the input data to the RNN model
# To fit the batches in the dataset, use data up until the batch size fits in the total length
def my_collate(batch):
    batch = list(filter(lambda img: img is not None, batch))
    transposed = zip(*batch)
    return [torch.utils.data.dataloader.default_collate(samples) for samples in transposed]


def load_data(batch_size):
    #features_train = torch.tensor(train[['RXMIN','RXMAX','FREQ','DIST']].values).type(torch.DoubleTensor)
    #features_test = torch.tensor(test[['RXMIN','RXMAX','FREQ','DIST']].values).type(torch.DoubleTensor)
    #features_val = torch.tensor(val[['RXMIN','RXMAX','FREQ','DIST']].values).type(torch.DoubleTensor)

    features_train = torch.tensor(train[['RXMIN', 'RXMAX']].values).type(torch.DoubleTensor)
    features_test = torch.tensor(test[['RXMIN', 'RXMAX']].values).type(torch.DoubleTensor)
    features_val = torch.tensor(val[['RXMIN', 'RXMAX']].values).type(torch.DoubleTensor)

    dates_train = torch.tensor(train['DATE'].values).type(torch.LongTensor)
    dates_test = torch.tensor(test['DATE'].values).type(torch.LongTensor)
    dates_val = torch.tensor(val['DATE'].values).type(torch.LongTensor)

    targets_train = torch.from_numpy(train['TARG_PRCP'].values).type(torch.DoubleTensor)
    targets_test = torch.from_numpy(test['TARG_PRCP'].values).type(torch.DoubleTensor)
    targets_val = torch.from_numpy(val['TARG_PRCP'].values).type(torch.DoubleTensor)

    # Transform the target data using a log transform. Add 1 to avoid mathematical errors when taking the logarithm of zero
    targets_train_transform = np.log10(targets_train+1)
    targets_test_transform  = np.log10(targets_test+1)
    targets_val_transform = np.log10(targets_val + 1)

    # Normalize the target data w.r.t. the full range of the data
    targets_test_norm = (targets_test_transform - targets_test_transform.min()) / (targets_test_transform.max() - targets_test_transform.min())
    targets_train_norm = (targets_train_transform - targets_train_transform.min()) / (targets_train_transform.max() - targets_train_transform.min())
    targets_val_norm = (targets_val_transform - targets_val_transform.min()) / (targets_val_transform.max() - targets_val_transform.min())

    features_train_norm = (features_train - features_train.min()) / (features_train.max() - features_train.min())
    features_test_norm = (features_test - features_test.min()) / (features_test.max() - features_test.min())
    features_val_norm = (features_val - features_val.min()) / (features_val.max() - features_val.min())

    train_pytorch = TensorDataset(features_train_norm,targets_train_norm)
    test_pytorch  = TensorDataset(features_test_norm,targets_test_norm)
    val_pytorch = TensorDataset(features_val_norm, targets_val_norm)

    return train_pytorch, test_pytorch, val_pytorch, dates_train, dates_test, dates_val


def DatesCheck(dates):

    DateCheck = True
    diffs = [y - x for x,y in zip(dates,dates[1:])]
    diffs = torch.stack(diffs).data.detach().numpy()

    # Month check
    month_nr = int(str(dates[1].item())[4:6])
    month_diff = 0
    if month_nr in [1, 3, 5, 7, 8, 10, 12]: month_diff = 697655
    elif month_nr in [4, 6, 9, 11]: month_diff = 707655
    else:
        if int(str(dates[1].item())[0:4]) % 4 == 0: month_diff = 717655 # Check for leap years
        else: month_diff = 727655
    acceptable_val = np.array([15,55,7655, month_diff,88697655]) # Only allow time gaps of 15 (a quarter diff), 55 (hour change), 7655 (day change)
                                                                 # month_diff (month change) and 88697655 (year change)

    # CHECK: is the time difference not a different number of minutes
    time_check = np.isin(diffs,acceptable_val)
    if sum(time_check) != len(time_check): DateCheck = False

    return DateCheck


class MyTrainDataset(torch.utils.data.Dataset):
        def __init__(self, seq_len=4, batch_size=64):
            self.traindata,_,_,self.traindates,_,_ = load_data(batch_size)
            self.seq_len = seq_len

        def __getitem__(self, item):
            if item < (self.seq_len-1):
                return None
            elif not DatesCheck(self.traindates[item-(self.seq_len-1):item+1]):
                return None
            else:
                return self.traindata.tensors[0][item - (self.seq_len - 1):item+1][0:4], self.traindata.tensors[1][item]

        def __len__(self):
            return len(self.traindata) - self.seq_len


class MyTestDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len=4, batch_size=64):
        _,self.testdata,_,_,self.testdates,_ = load_data(batch_size)
        self.seq_len = seq_len

    def __getitem__(self, item):
        if item < (self.seq_len-1):
            return None
        elif not DatesCheck(self.testdates[item - (self.seq_len - 1):item + 1]):
            return None
        else:
            return self.testdata.tensors[0][item - (self.seq_len - 1):item+1], self.testdata.tensors[1][item]

    def __len__(self):
        return len(self.testdata) - self.seq_len

class MyValDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len=4, batch_size=64):
        _,_,self.valdata,_,_,self.valdates = load_data(batch_size)
        self.seq_len = seq_len

    def __getitem__(self, item):
        if item < (self.seq_len-1):
            return None
        elif not DatesCheck(self.valdates[item - (self.seq_len - 1):item + 1]):
            return None
        else:
            return self.valdata.tensors[0][item - (self.seq_len - 1):item+1], self.valdata.tensors[1][item]

    def __len__(self):
        return len(self.valdata) - self.seq_len



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

    trainset = MyTrainDataset(seq_len=4, batch_size=batch_size)
    testset = MyTestDataset(seq_len=4, batch_size=batch_size)
    valset = MyValDataset(seq_len=4, batch_size=batch_size)

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn = my_collate, shuffle=False, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn = my_collate, shuffle=False, drop_last=True)

    loss_list = []
    iteration_list = []
    MSE_list = []
    count = 0
    MSE = 0
    for epoch in range(config['num_epochs']):
        for i, (features, targets) in enumerate(train_loader):
            train_set = torch.autograd.Variable(features)
            targets = torch.autograd.Variable(targets)
            train_set, targets = train_set.to(device), targets.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train_set)

            # Calculate softmax and MSE Loss
            loss = error(outputs, targets)

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

                # Iterate through test dataset
                for i, (features, targets) in enumerate(test_loader):
                    # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
                    test_set = torch.autograd.Variable(features)
                    targets = torch.autograd.Variable(targets)
                    test_set, targets = test_set.to(device), targets.to(device)

                    # Forward propagation
                    outputs = model(test_set)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]  # Useful when you're looking at a probability distribution

                    # Calculate MSE per batch
                    MSE_per_batch.append((outputs - targets).square().mean().item())
                MSE = np.average(MSE_per_batch)

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                MSE_list.append(MSE)

                if count % 800 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  MSE: {} %'.format(count, loss.data.item(), MSE))

                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=MSE)
    print('Finished training')


def val_accuracy(model, device='cpu'):
    valset = MyValDataset(seq_len=4, batch_size=batch_size)
    val_loader = DataLoader(valset, batch_size=batch_size,collate_fn=my_collate, shuffle=False,drop_last=True)
    MSE_per_batch = []
    with torch.no_grad():
        for i, (features,targets) in enumerate(val_loader):
            val_set = torch.autograd.Variable(features)
            labels = torch.autograd.Variable(targets)
            val_set, targets = val_set.to(device), targets.to(device)
            # Forward propagation
            outputs = model(val_set)

            MSE_per_batch.append((outputs - targets).square().mean().item())
            # accuracy = 100 * correct / float(total)
        mse = np.average(MSE_per_batch)
    return mse

def main(num_samples = 4, gpus_per_trial=0):

    config = {
        #'batchsize': tune.choice([16, 32, 64, 128, 256]),
        'lr': tune.loguniform(1e-6, 1e-3),
        'num_epochs': tune.randint(4, 25),
        'hidden_size': tune.choice([64,128,256]),#tune.choice([16, 32, 64, 128, 256]),
        'num_layers': tune.choice([2,4,6,8,16]),#tune.choice([2, 4, 8])
    }
    load_data(batch_size)
    scheduler = ASHAScheduler(metric='loss',mode='min',grace_period=1,reduction_factor=2)
    reporter = CLIReporter(metric_columns=['loss'])
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
    # best_trained_model.load_state_dict(model_state)
    test_acc = val_accuracy(best_trained_model,device)
    print("Best trial test set loss: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here and the number of samples taken:
    main(num_samples=25)




#################
# OLD MODEL RUN # This model run works, beforehand is the optimizing tuning algorithm implemented
#################
writer = SummaryWriter(filename_suffix='without FREQ and DIST 15 epochs')

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

train_pytorch = MyTrainDataset(seq_len=4, batch_size=64)
test_pytorch = MyTestDataset(seq_len=4, batch_size=64)
val_pytorch = MyValDataset(seq_len=4, batch_size=64)

train_loader = DataLoader(train_pytorch, batch_size=batch_size, collate_fn=my_collate, shuffle=False,drop_last=True)
test_loader = DataLoader(test_pytorch, batch_size=batch_size, collate_fn=my_collate, shuffle=False, drop_last=True)
val_loader = DataLoader(val_pytorch, batch_size=batch_size, collate_fn=my_collate, shuffle=False, drop_last=True)

for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(train_loader):
        train_set = torch.autograd.Variable(features)
        targets = torch.autograd.Variable(targets)

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

        count += 1

        if count % 400 == 0:
            # Calculate Accuracy
            print('count is: ', count)
            correct = 0
            MSE_per_batch = []
            MSE_per_batch_nontrans = []
            total = 0

            # Keep track of where you are in the test loop
            test_count = 0
            # Iterate through test dataset
            for features,targets in test_loader:
                # images = torch.autograd.Variable(images.view(-1, sequence_length, input_size))
                test_set = torch.autograd.Variable(features)
                targets = torch.autograd.Variable(targets)

                # Forward propagation
                outputs = model(test_set)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]  # Useful when you're looking at a probability distribution
                targets_nontrans = np.power(10,targets) - 1
                outputs_nontrans = np.power(10,outputs.detach().numpy().reshape(1,-1)) - 1

                MSE_per_batch.append((targets - outputs).square().mean().item())
                MSE_per_batch_nontrans.append((targets_nontrans - outputs_nontrans).square().mean().item())
                # accuracy = 100 * correct / float(total)
            MSE = np.average(MSE_per_batch)
            MSE_nontrans = np.average(MSE_per_batch_nontrans)
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            MSE_list.append(MSE)
            writer.add_scalar('Test loss', MSE, count)
            writer.add_scalar('Training loss',loss.data.item(),count)
            writer.add_scalar('Test loss (untransformed)',MSE_nontrans,count)
            if count % 800 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  MSE: {} %'.format(count, loss.data.item(), MSE))

writer.close()

#plt.imshow(target_data_nc_prcp, vmin=0, vmax=0.05)
#plt.colorbar()
#plt.show()

class MyDaatset(torch.utils.data.Dataset):
    def __init__(self, input, seq_len):
        self.input = input
        self.seq_len = seq_len
    def __getitem__(self, item):
        if item < self.seq_len:
            print(item)
            return None
        else: return self.input.tensors[0][item-self.seq_len:item], self.input.tensors[1][item]
    def __len__(self):
        return len(self.input) - self.seq_len

def my_collate(batch):
    print(type(batch))
    batch = list(filter(lambda img: img is not None, batch))
    transposed = zip(*batch)
    #return torch.utils.data.dataloader.default_collate(batch)
    return [torch.utils.data.dataloader.default_collate(samples) for samples in transposed] #(torch.utils.data.dataloader.default_collate

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