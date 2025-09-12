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

def build_dataloader(cfg, first_use_dataset):

    # Load dataset params for WSJ0 subset
    train_data_dir = cfg.get('User', 'train_data_dir')
    val_data_dir = cfg.get('User', 'val_data_dir')
    sample_mean = cfg.get('User','mean_file')
    sample_std = cfg.get('User','std_file')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    data_suffix = cfg.get('DataFrame', 'suffix')
    # feattype = cfg.get('STFT','feattype')
    # use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Load STFT parameters
    # wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    # zp_percent = cfg.getint('STFT', 'zp_percent')
    # wlen = np.int64(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    # nfft = cfg.getint('STFT','nfft')
    # win = 'hann'
    trim = cfg.getboolean('STFT', 'trim')

    # STFT_dict = {}
    # STFT_dict['fs'] = fs
    # STFT_dict['wlen'] = wlen
    # STFT_dict['hop'] = hop
    # STFT_dict['nfft'] = nfft
    # STFT_dict['win'] = win
    # STFT_dict['trim'] = trim

    # List all available speech audio
    if train_data_dir.endswith('.txt'):
        train_file_list = []
        with open(train_data_dir, 'r') as file:
            for line in file:
                # Strip any trailing whitespace, including newline characters
                stripped_line = line.rstrip()
        
                # Check if the line ends with the specified suffix
                if stripped_line.endswith('.wav'):
                    train_file_list.append(stripped_line)
        val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
    else:
        train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
        val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
    # train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
    # val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)

    # Training dataset
    train_dataset = SpeechSequencesFull(file_list=train_file_list, sequence_len=sequence_len, fs=fs, hop=hop, trim=trim,
                                         shuffle=shuffle, sample_mean=sample_mean, sample_std=sample_std, 
                                         name=dataset_name, first_use=first_use_dataset, dataset_to="train")
    val_dataset = SpeechSequencesFull(file_list=val_file_list, sequence_len=sequence_len, fs=fs, hop=hop, trim=trim,
                                     shuffle=shuffle, sample_mean=sample_mean, sample_std=sample_std, 
                                     name=dataset_name, first_use=first_use_dataset, dataset_to="val")
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
    def __init__(self, file_list, sequence_len, fs, hop, trim, shuffle, sample_mean, sample_std, name='WSJ0', first_use=True, dataset_to="train"):

        super().__init__()

        # STFT parameters
        self.fs = fs
        self.hop = hop
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle
        self.trim = trim
        # self.sample_mean = np.loadtxt(sample_mean)

        # self.sample_std = np.loadtxt(sample_std)
        # self.input_cfg = input_cfg

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

        for wavfile in self.file_list:

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
        # x1, fs_x = sf.read(wavfile)
        x, fs_x = librosa.load(wavfile, sr=None)

        # Sequence tailor
        x = x[seq_start:seq_end] # time domain signal

        # sample_lps_nor, frame_number, bin_number = extract_dns_lps(x, self.sample_mean, self.sample_std, self.wlen, self.nfft, self.hop, self.input_cfg)

        return x


                



