#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
random.seed(123)
import numpy as np
np.random.seed(123)
import soundfile as sf
import librosa
import torch
from torch.utils import data
import pickle

def build_dataloader_nsvae(cfg, first_use_dataset):

    # Load dataset params for WSJ0 subset
    noisy_train_data_dir = cfg.get('User', 'noisy_train_data_dir')
    clean_train_data_dir = cfg.get('User', 'clean_train_data_dir')
    noise_train_data_dir = cfg.get('User', 'noise_train_data_dir')

    noisy_val_data_dir = cfg.get('User', 'noisy_val_data_dir')
    clean_val_data_dir = cfg.get('User', 'clean_val_data_dir')
    noise_val_data_dir = cfg.get('User', 'noise_val_data_dir')

    dataset_name = cfg.get('DataFrame', 'dataset_name')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    data_suffix = cfg.get('DataFrame', 'suffix')
    # use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Load STFT parameters
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    trim = cfg.getboolean('STFT', 'trim')



    # List all available speech audio
    if noisy_train_data_dir.endswith('.txt'):
        noisy_train_file_list = []
        with open(noisy_train_data_dir, 'r') as file:
            for line in file:
                # Strip any trailing whitespace, including newline characters
                stripped_line = line.rstrip()
        
                # Check if the line ends with the specified suffix
                if stripped_line.endswith('.wav'):
                    noisy_train_file_list.append(stripped_line)
        noisy_val_file_list = librosa.util.find_files(noisy_val_data_dir, ext=data_suffix)
    else:
        noisy_train_file_list = librosa.util.find_files(noisy_train_data_dir, ext=data_suffix)
        noisy_val_file_list = librosa.util.find_files(noisy_val_data_dir, ext=data_suffix)

    # clean_train_file_list = librosa.util.find_files(clean_train_data_dir, ext=data_suffix)
    # clean_val_file_list = librosa.util.find_files(clean_val_data_dir, ext=data_suffix)

    # noise_train_file_list = librosa.util.find_files(noise_train_data_dir, ext=data_suffix)
    # noise_val_file_list = librosa.util.find_files(noise_val_data_dir, ext=data_suffix)

    # Training dataset
    train_dataset = SpeechSequencesFull(noisy_file_list=noisy_train_file_list, clean_file_dir=clean_train_data_dir, noise_file_dir=noise_train_data_dir, 
                                        sequence_len=sequence_len, hop=hop, fs=fs, trim=trim, shuffle=shuffle, 
                                         name=dataset_name, first_use=first_use_dataset, dataset_to='train')

    val_dataset = SpeechSequencesFull(noisy_file_list=noisy_val_file_list, clean_file_dir=clean_val_data_dir, noise_file_dir=noise_val_data_dir, 
                                        sequence_len=sequence_len, hop=hop, fs=fs, trim=trim, shuffle=shuffle, 
                                        name=dataset_name, first_use=first_use_dataset, dataset_to='val')


    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num


class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, noisy_file_list, clean_file_dir, noise_file_dir, sequence_len, hop, fs, trim, shuffle, name='WSJ0', first_use=True, dataset_to='train'):

        super().__init__()
        
        # data parameters
        self.noisy_file_list = noisy_file_list
        self.clean_file_dir = clean_file_dir
        self.noise_file_dir = noise_file_dir

        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle


        self.hop = hop
        self.fs = fs
        self.trim = trim
        if first_use:
            if dataset_to == "train":
                self.compute_len("train")
            elif dataset_to == "val":
                self.compute_len("val")
        else:
            if dataset_to == "train":
                with open(self.name + "_train.pkl", "rb") as file:
                    self.valid_seq_list = pickle.load(file)
            elif dataset_to == "val":
                with open(self.name + "_val.pkl", "rb") as file:
                    self.valid_seq_list = pickle.load(file)


    def compute_len(self, dataset_to):

        self.valid_seq_list = []

        for wavfile in self.noisy_file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            
            # remove beginning and ending silence

            if self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                ind_beg = 0
                ind_end = len(x)


            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)

        if dataset_to == "train":
            with open(self.name + "_train.pkl", "wb") as file:
                pickle.dump(self.valid_seq_list, file)

        if dataset_to == "val":
            with open(self.name + "_val.pkl", "wb") as file:
                pickle.dump(self.valid_seq_list, file)

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        fileid = wavfile.split('.')[0]
        fileid = fileid.split('_')[-1]
        clean_file = self.clean_file_dir + '/' + 'clean_fileid_' + fileid + '.wav'
        noise_file = self.noise_file_dir + '/' + 'noise_fileid_' + fileid + '.wav'
        # x1, fs_x = sf.read(wavfile)
        x, fs_x = librosa.load(wavfile, sr=None)
        clean_x, fs_x = librosa.load(clean_file, sr=None)
        noise_x, fs_x = librosa.load(noise_file, sr=None)

        # Sequence tailor
        x = x[seq_start:seq_end]
        clean_x = clean_x[seq_start:seq_end]
        noise_x = noise_x[seq_start:seq_end]
        
        return x, clean_x, noise_x



