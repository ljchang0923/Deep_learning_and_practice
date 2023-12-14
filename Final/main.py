import numpy as np
from scipy import signal
from EEG_VAE import vae
from EEG_AVAE import avae
from EEG_WGAN import wgan
from EEG_DCGAN import dcgan
from MI_VAE import MI_vae
from fft_plot import fft_data, plot_fft
from scipy import io
import os
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events, scale
from braindecode.augmentation import FrequencyShift, FTSurrogate, GaussianNoise
seed = 99


def normalize(data): 
    print(data.shape)
    means = np.mean(data, axis=2, keepdims=True)
    stds = np.std(data, axis=2, keepdims=True)
    normalized_data = (data - means) / stds
    
    return normalized_data

'''
for MAMEM2
'''
def read_data_with_label(subject_id, label_id):
    dataset = MOABBDataset(dataset_name="MAMEM1", subject_ids=[subject_id])

    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors 
        # Preprocessor(scale, factor=1e-1, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=0.5, h_freq=40),

    ]

    # Transform the data
    preprocess(dataset, preprocessors)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )

    # channel: O1->116 Oz->126 O2->150  
    X_arr = []
    Y_arr = []
    for i in range(len(windows_dataset.datasets)):
        for j in range(len(windows_dataset.datasets[i])):
            X_arr.append(windows_dataset.datasets[i][j][0])
            Y_arr.append(windows_dataset.datasets[i][j][1])

    X = np.stack(X_arr)    
    Y = np.array(Y_arr)
    X = np.stack((X[:,115,:],X[:,125,:],X[:,149,:]),axis=1)
    # X.shape = (125, 3, 750)
    # Y.shape = (125)
    X_label = []
    Y_label = []
    for i in range(X.shape[0]):
        if Y[i] == label_id:
            X_label.append(X[i])
            Y_label.append(Y[i])
    X_label = np.stack(X_label)
    Y_label = np.stack(Y_label)

    X_label = normalize(X_label)

    return X_label, Y_labelxx


def DA_with_fn(X, Y, number):
    transform = FTSurrogate(probability=1)
    shift_hz = 5
    sfreq = 250
    # transform = FrequencyShift(probability=1, sfreq=sfreq)
    transform = GaussianNoise(probability=1)
    
    while(X.shape[0]<number):
        augmented_X, _ = transform.operation(torch.as_tensor(X).float(), None, 1)
        # augmented_X, _ = transform.operation(torch.as_tensor(X).float(), None, None, sfreq)
        X = np.concatenate((augmented_X, X), axis=0)
        Y = np.concatenate((Y, Y), axis=0)
    X = X[:number]
    Y = Y[:number]

    return X, Y


def plot_tsne(x, y, org_gen, subject):
    data = []
    for trial in x:
        data.append(trial.reshape(-1))
    
    data = np.stack(data)

    X_tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=3000).fit_transform(data)
    print(f"embedded shape: {X_tsne.shape}")

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

    colors = cm.Set1(np.linspace(0, 1, 2))
    plt.figure(figsize=(8,8))
    plt.scatter(X_norm[:,0], X_norm[:,1], 10, org_gen, alpha=0.5)
    # # plt.scatter(X_norm[0:int(5*len(X_norm)/6), 0],X_norm[0:int(5*len(X_norm)/6), 1], 10 ,colors[0])
    # # plt.scatter(X_norm[int(5*len(X_norm)/6):, 0],X_norm[int(5*len(X_norm)/6):, 1], 10 ,colors[1])
    
    plt.legend(["original", "generated"], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"fig_gau/sub{subject}_t-sne_plot_org_gen.png")
    plt.clf()

    colors = cm.rainbow(np.linspace(0, 1, 5))
    plt.figure(figsize=(8,8))
    for c in range(5):
        mask = (y==c)
        x_c = X_norm[mask]
        plt.scatter(x_c[:,0], x_c[:,1], 10, colors[c])
    
    plt.legend(["class0", "class1", "class2", "class3", "class4"], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"fig_gau/sub{subject}_t-sne_plot_class_test.png")
    plt.clf()

if __name__=="__main__":
    T=3
    for sub in range(1,6):
        subject = sub
        org_data = []
        org_label = []
        gen_data = []
        gen_label = []
        for c in range(5):
            X, Y = read_data_with_label(subject, c)
            org_data.append(X[:int(len(X)*0.3)])
            org_label.append(Y[:int(len(X)*0.3)])
            # print("X max min: ", np.max(X),"\t" ,np.min(X))
            # gen_data_class = wgan(X[int(len(X)*0.3):], Y[int(len(X)*0.3):], seed)
            gen_data_class, _ = DA_with_fn(X[int(len(X)*0.3):], Y[int(len(X)*0.3):], 500)
            gen_data.append(gen_data_class)
            gen_data.append(X[int(len(X)*0.3):])
            gen_label.append(np.repeat(c,len(gen_data_class)))
            gen_label.append(Y[int(len(X)*0.3):])
            gen_data_class = gen_data_class.swapaxes(1,2)

            data_fft = fft_data(gen_data_class)
            x_axis = np.linspace(0,(gen_data_class.shape[1]/T)/2, data_fft.shape[0])
            plot_fft(x_axis, data_fft, f"fig_gau/sub{subject}_gen_class{c}_fft")

            # FFT
            X = X.swapaxes(1,2)
            data_fft = fft_data(X)
            x_axis = np.linspace(0,(X.shape[1]/T)/2, data_fft.shape[0])
            plot_fft(x_axis, data_fft, f"fig_gau/sub{subject}_raw_class{c}_fft")

        org_data = np.concatenate(org_data)
        org_label = np.concatenate(org_label)
        gen_data = np.concatenate(gen_data)
        gen_label = np.concatenate(gen_label)
        all_data = np.concatenate([org_data, gen_data])
        all_label = np.concatenate([org_label, gen_label])
        all_label2 = np.concatenate([np.repeat(0,len(org_label)), np.repeat(1, len(gen_label))])
        print("size of label: ", all_label.shape)
        with open(f'fig_gau/sub{subject}_data.npy', 'wb') as f:
            np.save(f, all_data)
        with open(f'fig_gau/sub{subject}_label.npy', 'wb') as f:
            np.save(f, all_label)
        with open(f'fig_gau/sub{subject}_label2.npy', 'wb') as f:
            np.save(f, all_label2)

        plot_tsne(all_data, all_label, all_label2, subject)
