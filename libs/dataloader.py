import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from einops import rearrange

import torch
from torch.utils.data import TensorDataset, DataLoader


def process_indian_pines_dataset(hsi, label):
    hsi_vectorized = rearrange(hsi, 'm n l -> (m n) l').astype(float)
    label_vectorized = rearrange(label, 'm n -> (m n)')

    remove_indices = label_vectorized == 0
    hsi_vectorized = hsi_vectorized[~remove_indices]
    label_vectorized = label_vectorized[~remove_indices] - 1

    for band in range(hsi_vectorized.shape[-1]):
        band_min = hsi_vectorized[:, band].min()
        band_max = hsi_vectorized[:, band].max()
        hsi_vectorized[:, band] = (hsi_vectorized[:, band] - band_min) / (band_max - band_min)

    return hsi_vectorized, label_vectorized


def split_dataset(spec, label, split, seed):
    np.random.seed(seed)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for lab in range(np.max(label) + 1):
        label_indices = np.argwhere(label == lab).squeeze()
        np.random.shuffle(label_indices)

        split_train = int(split['train'] * len(label_indices))
        x_train.extend(spec[label_indices[:split_train]])
        x_test.extend(spec[label_indices[split_train:]])

        y_train.extend(label[label_indices[:split_train]])
        y_test.extend(label[label_indices[split_train:]])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def normalize_data_by_row(data):
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        row = data[i, :]
        
        # Encontrar el valor máximo en la fila
        max_value = np.max(row)
        
        # Normalizar la fila
        if max_value > 0:  # Prevenir división por cero
            normalized_data[i, :] = row / max_value

    return normalized_data


def normalize_data_min_max_by_row(data):
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        row = data[i, :]
        
        # Encontrar el valor mínimo y máximo en la fila
        min_value = np.min(row)
        max_value = np.max(row)
        
        # Rango de la fila
        range_value = max_value - min_value
        
        # Normalizar la fila
        if range_value > 0:  # Prevenir división por cero
            normalized_data[i, :] = (row - min_value) / range_value

    return normalized_data


def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-5

    X_train_standardized = (X_train - mean) / std
    X_test_standardized  = (X_test  - mean) / std

    return X_train_standardized, X_test_standardized

# apply multiplicative scatter correction
def msc_fn(x, ref_signal):
    """
    Perform Multiplicative Scatter Correction (MSC) on a given vector.

    MSC is used to correct spectral data by removing scatter effects. The correction is 
    applied using the formula: x_corr = (x - a) / b, where 'a' and 'b' are calculated 
    based on the reference signal.

    Parameters:
    x (numpy.ndarray): The vector to be corrected.
    ref_signal (numpy.ndarray): The reference signal of the dataset.

    Returns:
    numpy.ndarray: The corrected vector.

    """
    x = x.squeeze()
    ref_signal = ref_signal.squeeze()

    mean_ref = np.mean(ref_signal)

    b = np.sum( ( x - x.mean() ) * ( ref_signal - mean_ref ) )  
    b = b / np.sum( (ref_signal - mean_ref)**2 )
    a = np.mean(x) - np.mean(ref_signal) * b
    
    x_corr = (x - a) / b

    return x_corr

def prepare_data(dataset_name, split, seed, dl=False, dataset_params=None, msc=True, device="cpu"):
    if dataset_name == 'indian_pines':
        data_path = 'data/indian_pines'
        hsi = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['hyperimg']
        label = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['hyperimg_gt']
        spec, label = process_indian_pines_dataset(hsi, label)

    elif dataset_name == 'cocoa_public':
        data = sio.loadmat('data/cocoa_public/cocoa_public.mat')
        spec = data['data']
        label = data['label'].squeeze()

    elif dataset_name == 'cocoa_public_pca':
        data = sio.loadmat('data/trash/cocoa_public_pca.mat')
        spec = data['data']
        label = data['label'].squeeze()
    
    elif dataset_name == 'closed_automatic':

        data_path = os.path.join('.', 'data', 'closed_automatic')

        data_path_train = os.path.join(data_path, 'train_cocoa_dataset.h5')
        data_path_test  = os.path.join(data_path, 'test_cocoa_dataset.h5')

        labels_path = os.path.join(".", "data", 'labels.csv')
        labels_pd = pd.read_csv(labels_path)
        cat_name = "ferm"

        labels = dict(zip(labels_pd['id'], labels_pd[cat_name]))           

        with h5py.File(data_path_train, 'r') as f:
            x_train = f['spec'][:]
            y_train = f['label'][:]

            print(y_train.shape)
            y_train = np.array([labels[int(i)] for i in y_train])[..., np.newaxis]
            
        
        with h5py.File(data_path_test, 'r') as f:

            x_test = f['spec'][:]
            y_test = f['label'][:]
            y_test = np.array([labels[int(i)] for i in y_test])[..., np.newaxis]
        
        y_train = y_train.astype(np.float32) / 100
        y_test  = y_test.astype(np.float32) / 100


    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # -----
    # replace this line when dataset = closed_automatic to load the default dataset split

    # if msc:
    #     ref_signal = np.mean(x_train, axis=0)

    #     for i in range(x_train.shape[0]):
    #         x_train[i, :] = msc_fn(x_train[i, :], ref_signal)
        
    #     for i in range(x_test.shape[0]):
    #         x_test[i, :]  = msc_fn(x_test[i, :], ref_signal)

    if dataset_name != 'closed_automatic':
        (x_train, y_train), (x_test, y_test) = split_dataset(spec, label, split, seed)
    # -----

    # x_train, x_test = standardize_data(x_train, x_test)

    num_bands = x_train.shape[-1]
    num_classes = len(np.unique(y_train))

    print("==============================================")
    print("x_train | shape = {} | type = {}".format(x_train.shape, x_train.dtype))
    print("y_train | shape = {} \t  | type = {}".format(y_train.shape, y_train.dtype))
    print("x_test  | shape = {} | type = {}".format(x_test.shape, x_test.dtype))
    print("y_test  | shape = {} \t  | type = {}".format(y_test.shape, y_test.dtype))
    print("Number of bands   = {}".format(num_bands))
    print("Number of classes = {}".format(num_classes))
    print("==============================================")

    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))

    if dl:  # deep learning dataset
        x_train = torch.from_numpy(x_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        train_dataset = TensorDataset(x_train, y_train)

        x_test = torch.from_numpy(x_test).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)
        
        test_dataset = TensorDataset(x_test, y_test)

        pin_param = True if device == "cpu" else False

        persistent_workers = True if dataset_params["num_workers"] > 0 else False
        train_dataset = DataLoader(train_dataset, batch_size=dataset_params["batch_size"], shuffle=True,
                                   num_workers=dataset_params["num_workers"], pin_memory=pin_param,
                                   persistent_workers=persistent_workers)
        
        test_dataset = DataLoader(test_dataset, batch_size=dataset_params["batch_size"], shuffle=False,
                                  num_workers=dataset_params["num_workers"], pin_memory=pin_param,
                                  persistent_workers=persistent_workers)

    else:  # machine learning dataset
        train_dataset = dict(X=x_train.astype(np.float64), Y=y_train.astype(np.uint8))
        test_dataset = dict(X=x_test.astype(np.float64), Y=y_test.astype(np.uint8))
        
    return train_dataset, test_dataset, num_bands, num_classes
