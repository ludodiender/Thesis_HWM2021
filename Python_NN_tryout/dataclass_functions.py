import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path


def my_collate(batch):
    batch = list(filter(lambda img: img is not None, batch))
    transposed = zip(*batch)
    return [torch.utils.data.dataloader.default_collate(samples) for samples in transposed]


def load_data_separate(data_input, do_transform = True,include_dist_as_feature = False):
    dates = torch.tensor(data_input['DATE'].values).type(torch.LongTensor)
    if len(dates) == 0:
        print('Dates are empty!')

    if do_transform:
        features_div = data_input[['RXMIN', 'RXMAX']].divide(data_input['DIST'], axis=0)
        features = torch.tensor(features_div.values)
        # print('Size of features:', features.size())
        # features_dist = features / data_input['DIST'] # switch to distance scaled data


        # Check which pair of indices belongs to the data to scale in the right way
        minmax_values = [[91.0, -109.0, 80.0, -127.0, 170.1999969482422, 0.0],
                         [63.0, -120.0, 53.0, -120.0, 107.91999053955078, 0.0],
                         [-22.0, -116.0, -23.0, -120.0, 55.959999084472656, 0.0]]
        mean_std_values = [[-49.473930381146616, 6.060946615463768, -50.382829844984684, 6.344096237772375],
                           [-49.53695990531813, 6.323105028045907, -50.43677002865799, 6.550715358080604],
                           [-49.63890282495527, 7.190955243365654, -50.41163702248873,
                            7.328972371162886]]  # From mean_std_perdataset.txt

        mean_std_values_dist = [[-34.035598769847695, 43.85211474505614, -34.62937509042944, 44.69098481523913],
                                [-34.015993618748375, 39.15742736484588, -34.52800696427108, 39.71461401889631],
                                [-33.58378801837597, 39.27559623712495, -34.02705358359594, 39.77195550629715]]
        index = 2
        if str(dates[0].item())[0:4] == '2011':
            index = 0
        elif str(dates[0].item())[0:4] == '2012':
            index = 1

        RXMAX_max, RXMAX_min, RXMIN_max, RXMIN_min, targ_max, targ_min = minmax_values[index]
        RXMAX_mean, RXMAX_std, RXMIN_mean, RXMIN_std = mean_std_values_dist[index]  # Switch to distance scaled data
        targets = torch.from_numpy(data_input['TARG_PRCP'].values).type(torch.DoubleTensor)

        # Check for NaNs in the features or targets
        if np.count_nonzero(np.isnan(features.detach().numpy())) != 0:
            print('Features already NaN in loading the data from file!')
            print('NaN features found in Link ID', data_input['ID'].iloc[0])

        if np.count_nonzero(np.isnan(targets.detach().numpy())) != 0:
            print('Targets already NaN in loading the data from file!')
            print('NaN targets found in Link ID', data_input['ID'].iloc[0])

        # Transform target data using a log sine transform. Add 1 to avoid mathematical errors when taking the logarithm of zero
        targets_transform = np.log10(targets + np.sqrt(np.power(targets, 2) + 1))
        targ_max_transform = np.log10(targ_max + np.sqrt(np.power(targ_max, 2) + 1))
        targ_min_transform = np.log10(targ_min + np.sqrt(np.power(targ_min, 2) + 1))

        # Normalize the target data w.r.t. the full range of the data
        if targets_transform.min() == targets_transform.max():
            targets_norm = targets_transform
        else:
            targets_norm = (targets_transform - targ_min_transform) / (targ_max_transform - targ_min_transform)

        # Normalize the features as well using mean and standard deviation scaling
        # mean_tensor = torch.tensor([RXMIN_mean, RXMAX_mean])
        median_tensor = torch.tensor(
            [np.median(features_div['RXMIN']), np.median(features_div['RXMAX'])])  # Calculate the median for each link
        std_tensor = torch.tensor([RXMIN_std, RXMAX_std])

        # features_norm = (features - mean_tensor) / (std_tensor)  # Switch to distance scaled data here as well
        features_norm = (features - median_tensor) / (std_tensor)  # Normalize by using the median

        if include_dist_as_feature:
            features_norm = torch.cat((features_norm, torch.tensor(data_input['DIST'].values).unsqueeze(1)), dim=1)

        # print('Size of target tensor:',targets.unsqueeze(0).size())
        # Check for NaNs after the transformation
        if np.count_nonzero(np.isnan(targets_norm.detach().numpy())) != 0:
            print('Targets become NaN after transforming!')
            print('NaN targets found in Link ID', data_input['ID'].iloc[0])

        if np.count_nonzero(np.isnan(features_norm.detach().numpy())) != 0:
            print('Features become NaN in transforming!')
            print('Feature min:', features.min(), ' Features max:', features.max())
            print('NaN features found in Link ID', data_input['ID'].iloc[0])
    else:
        features_norm = torch.from_numpy(data_input[['RXMIN','RXMAX']].values).type(torch.DoubleTensor)
        if include_dist_as_feature:
            features_norm = torch.cat((features_norm, torch.tensor(data_input['DIST'].values).unsqueeze(1)), dim=1)

        targets_norm = torch.from_numpy(data_input['TARG_PRCP'].values).type(torch.DoubleTensor)


    # Combine the features and the targets in one Tensor Dataset
    data_pytorch = TensorDataset(features_norm, targets_norm)

    return data_pytorch, dates


def DatesCheck(dates):
    DateCheck = True
    diffs = [y - x for x, y in zip(dates, dates[1:])]
    if len(diffs) == 0:
        print('Diffs is empty!')
        DateCheck = False

    else:
        diffs = torch.stack(diffs).data.detach().numpy()

        # Month check
        month_nr = int(str(dates[1].item())[4:6])
        month_diff = 0
        if month_nr in [1, 3, 5, 7, 8, 10, 12]:
            month_diff = 697655
        elif month_nr in [4, 6, 9, 11]:
            month_diff = 707655
        else:
            if int(str(dates[1].item())[0:4]) % 4 == 0:
                month_diff = 717655  # Check for leap years
            else:
                month_diff = 727655
        acceptable_val = np.array([15, 55, 7655, month_diff,
                                   88697655])  # Only allow time gaps of 15 (a quarter diff), 55 (hour change), 7655 (day change)
        # month_diff (month change) and 88697655 (year change)

        # CHECK: is the time difference not a different number of minutes
        time_check = np.isin(diffs, acceptable_val)
        if sum(time_check) != len(time_check): DateCheck = False

    return DateCheck


def get_sample_count_by_file(path: Path) -> int:
    c = -1  # Start at minus 1 since the Header is included in the csv file, but not when it is loaded as a dataframe
    with path.open() as f:
        for line in f:
            c += 1
    return c

def custom_MSE_RDF_loss(outputs, targets):
    ys = 0.95
    yr = -5
    new_outputs = torch.mul(targets.double(),yr)
    RDF = torch.where(outputs.double() < 0, 2.0, (1 - (ys * torch.exp(new_outputs))))
    MSE = ((outputs.double() - targets.double())**2)
    loss = torch.mean(RDF * MSE)
    if loss.item() < 0:
        print('Output tensor:',outputs,'Target tensor:',targets)
    return loss