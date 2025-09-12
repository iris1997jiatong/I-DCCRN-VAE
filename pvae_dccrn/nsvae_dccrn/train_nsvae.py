
from __future__ import print_function

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

torch.manual_seed(123)
torch.cuda.manual_seed(123)
# torch.manual_seed(333)
# torch.cuda.manual_seed(333)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(main_folder_path)

from model.pvae_module import *
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig
from model.nsvae_loss import standard_nsvae_loss_by_sampling, standard_nsvae_loss_true_kl
from dataset import dataload_nsvae
import datetime
import argparse
import socket
import os
import shutil
from utils.logger import get_logger
import pickle
from utils.read_config import myconf

import neptune
from neptune_pytorch import NeptuneLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def check_and_log_nan(tensor, name):
    if tensor is not None:
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"inf detected in {name}")
            raise RuntimeError(f"inf detected in {name}")

def beta_pvae(cfg, log_params):

    # get basic info
    date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
    hostname = socket.gethostname()
    model_name = cfg.get('User', 'model_name')
    dataset_name = cfg.get('DataFrame', 'dataset_name')


    basic_info = []
    basic_info.append('HOSTNAME: ' + hostname)
    basic_info.append('Time: ' + date)
    basic_info.append('model name:' + model_name)
    basic_info.append('Device for training: ' + device)
    if device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))

    # load training parameters
    epochs = cfg.getint('Training', 'epochs')
    early_stop_patience = cfg.getint('Training', 'early_stop_patience')

    learning_rate = cfg.getfloat('Training', 'lr')
    flag_clean_encoder = cfg.getboolean('Network','clean_encoder')
    flag_clean_decoder = cfg.getboolean('Network','clean_decoder')
    flag_noise_encoder = cfg.getboolean('Network','noise_encoder')
    flag_noise_decoder = cfg.getboolean('Network','noise_decoder')
    pre_clean_vae_en = cfg.get('User','pre_clean_encoder')
    pre_clean_vae_de = cfg.get('User','pre_clean_decoder')
    pre_noise_vae_en = cfg.get('User','pre_noise_encoder')
    pre_noise_vae_de = cfg.get('User','pre_noise_decoder')

    # load model
    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')    
    num_samples = log_params['num_samples']
    zdim = log_params['zdim']
    causal = log_params['causal']
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()
    # pretrain params
    setups = pre_clean_vae_en.split('/')[-2]
    if 'fcl' not in setups:
        fcl = False
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
    if 'spadd' not in setups:
        spadd = False
    setups = setups.split('_')
    for s in setups:
        if 'skipc' in s:
            skipc = s.split('=')[-1]
        elif 'skipuse' in s:
            tmp = s.split('=')[-1][1:-1]
            tmp = tmp.split(', ')
            skipuse = []
            for n in tmp:
                skipuse.append(int(n))
        elif 'recon=' in s:
              recon_type = s.split('=')[-1]
              if recon_type == 'real':
                    recon_type = 'real_imag'
        elif 'spadd' in s:
            spadd_tmp = s.split('=')[-1]
            spadd = (spadd_tmp.lower() == 'true')
        elif 'fcl=' in s:
            fcl = s.split('=')[-1]
            fcl = (fcl.lower() == 'true')
        
    print('fcl', fcl)
            
    if skipc == 'False':
        if not fcl:
            if not spadd:
                clean_model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
                noise_model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
            else:
                clean_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                clean_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)
                noise_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)   
        else:
            if not spadd:
                clean_model_encoder = pvae_dccrn_encoder_no_skip_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
                noise_model_encoder = pvae_dccrn_encoder_no_skip_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
            else:
                clean_model_encoder = pvae_dccrn_encoder_skip_prepare_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                clean_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)
                noise_model_encoder = pvae_dccrn_encoder_skip_prepare_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)   
                         
    else:
        clean_model_encoder = pvae_dccrn_encoder(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
        clean_model_decoder = pvae_dccrn_decoder(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse)            
        noise_model_encoder = pvae_dccrn_encoder(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
        noise_model_decoder = pvae_dccrn_decoder(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse)           

    # nsvae model params
    nsvae_model = log_params['nsvae_model']
    latent_num = log_params['latent_num']
    if not spadd:
        if nsvae_model == 'original':
            noisy_model_encoder = nsvae_dccrn_encoder_original(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, latent_num)
        elif nsvae_model == 'double':
            noisy_model_encoder = nsvae_dccrn_encoder_double_channel(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, latent_num)
        elif nsvae_model == 'adapt':
            noisy_model_encoder = nsvae_dccrn_encoder_adapt_channel(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, latent_num, skipuse) 
    else:
        if not fcl:
            noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, latent_num)
        else:
            noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, latent_num)
    basic_info.append('clean encoder params: %.2fM' % (sum(p.numel() for p in clean_model_encoder.parameters()) / 1000000.0))
    basic_info.append('noise encoder params: %.2fM' % (sum(p.numel() for p in noise_model_encoder.parameters()) / 1000000.0))
    basic_info.append('clean decoder params: %.2fM' % (sum(p.numel() for p in clean_model_decoder.parameters()) / 1000000.0))
    basic_info.append('noise decoder params: %.2fM' % (sum(p.numel() for p in noise_model_decoder.parameters()) / 1000000.0))
    basic_info.append('noisy encoder params: %.2fM' % (sum(p.numel() for p in noisy_model_encoder.parameters()) / 1000000.0))
    # load pretrained CVAE and NVAE
    clean_model_encoder.load_state_dict(torch.load(pre_clean_vae_en))
    clean_model_decoder.load_state_dict(torch.load(pre_clean_vae_de))

    noise_model_encoder.load_state_dict(torch.load(pre_noise_vae_en))
    noise_model_decoder.load_state_dict(torch.load(pre_noise_vae_de))

    # noisy_model_encoder.load_state_dict(torch.load(pre_clean_vae_en))

    clean_model_encoder.to(device)
    clean_model_decoder.to(device)
    noise_model_encoder.to(device)
    noise_model_decoder.to(device)
    noisy_model_encoder.to(device)
    #    noisy_model_decoder.to(device)

    # define loss
    alpha = log_params['alpha']
    w_resi = log_params['w_resi']
    w_kl = log_params['w_kl']
    matching = log_params['matching']
    w_dismiu = log_params['w_dismiu']
    # model_loss_sampling = standard_nsvae_loss_by_sampling(alpha, w_resi, w_kl, zdim, num_samples, latent_num, nsvae_model, skipc, skipuse, matching)
    model_loss = standard_nsvae_loss_true_kl(alpha, w_resi, w_kl, w_dismiu, zdim, num_samples, latent_num, nsvae_model, skipc, skipuse, matching)
    
    # model_loss.to(device)
    optimizer_noisy_en = optim.Adam(noisy_model_encoder.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler_noisy_en = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noisy_en, 'min', factor=0.5,patience=3)
    #    optimizer_noisy_de = optim.Adam(noisy_model_decoder.parameters(), lr=learning_rate, weight_decay=0.001)
    if flag_noise_encoder:
            optimizer_noise_en = optim.Adam(noise_model_encoder.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler_noise_en = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noise_en, 'min', factor=0.5,patience=3)
    else:
            for param in noise_model_encoder.parameters():
                    param.requires_grad = False
    if flag_noise_decoder:
            optimizer_noise_de = optim.Adam(noise_model_decoder.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler_noise_de = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noise_de, 'min', factor=0.5,patience=3)
    else:
            for param in noise_model_decoder.parameters():
                    param.requires_grad = False
    if flag_clean_encoder:
            optimizer_clean_en = optim.Adam(clean_model_encoder.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler_clean_en = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_clean_en, 'min', factor=0.5,patience=3)
    else:
            for param in clean_model_encoder.parameters():
                    param.requires_grad = False
    if flag_clean_decoder:
            optimizer_clean_de = optim.Adam(clean_model_decoder.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler_clean_de = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_clean_de, 'min', factor=0.5,patience=3)
    else:
            for param in clean_model_decoder.parameters():
                    param.requires_grad = False


    if not log_params['reload']:
            saved_root = cfg.get('User', 'saved_root')
            filename = "{}_{}_causal={}_zdim={}_alpha={:.2f}_wresi={:.1f}_wkl={}_wdismiu={}_numsamples={}_nsvae={}_latentnum={}_match={}".format(date, model_name, causal, zdim, alpha, w_resi, w_kl, w_dismiu, num_samples,
                                                                                                                   nsvae_model, latent_num, matching)
            save_dir = os.path.join(saved_root, filename)
            if not(os.path.isdir(save_dir)):
                    os.makedirs(save_dir)
            print(filename)
    else:
    #     tag = self.cfg.get('Network', 'tag')
        save_dir = log_params['model_dir']
        print(save_dir)

    # save the model configuration
    save_cfg = os.path.join(save_dir, 'config.ini')
    shutil.copy(log_params['cfg_file'], save_cfg)


    # initialize neptune loss logger
    # run = neptune.init_run(
    # project="ljt19970110/NSVAE",
    # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWFiNTU5YS1mMTFiLTQwNmEtYWI0YS0wZjlhMDJmZjY1ZWIifQ==",
    # )      
    # run["sys/description"] = filename 
    # run["aux/config"].upload(log_params['cfg_file']) 



    # create logger
    log_file = os.path.join(save_dir, 'log.txt')
    logger_type = cfg.getint('User', 'logger_type')
    logger = get_logger(log_file, logger_type)

    # Print basical infomation
    for log in basic_info:
            logger.info(log)
    logger.info('In this experiment, result will be saved in: ' + save_dir)

    logger.info('cvae encoder -- {}'.format(pre_clean_vae_en))
    logger.info('cvae decoder -- {}'.format(pre_clean_vae_de))
    logger.info('nvae encoder -- {}'.format(pre_noise_vae_en))
    logger.info('nvae decoder -- {}'.format(pre_noise_vae_de))      



    # load data
    first_use_dataset = log_params['first_use_dataset']
    train_dataloader, val_dataloader, train_num, val_num = dataload_nsvae.build_dataloader_nsvae(cfg, first_use_dataset)
    logger.info('Train on {}'.format(dataset_name))
    logger.info('Training samples: {}'.format(train_num))
    logger.info('Validation samples: {}'.format(val_num))  

    # Create python list for loss
    if not log_params['reload']:
            #train loss
            epoch_train_total_loss = np.zeros((epochs,))
        #   epoch_train_noisy_recon_loss = np.zeros((epochs,))
            epoch_train_noisy_kl_loss = np.zeros((epochs,))
            epoch_train_noisy_klclean_loss = np.zeros((epochs,))
            epoch_train_noisy_klnoise_loss = np.zeros((epochs,))
            epoch_train_noisy_dismiu_speech_loss = np.zeros((epochs,))
            epoch_train_noisy_dismiu_noise_loss = np.zeros((epochs,))
            epoch_train_noisy_residual_loss = np.zeros((epochs,))
            epoch_train_noisy_residual_loss_speech = np.zeros((epochs,))
            epoch_train_noisy_residual_loss_noise = np.zeros((epochs,))

            # val loss
            epoch_val_total_loss = np.zeros((epochs,))
        #   epoch_val_noisy_recon_loss = np.zeros((epochs,))
            epoch_val_noisy_kl_loss = np.zeros((epochs,))
            epoch_val_noisy_klclean_loss = np.zeros((epochs,))
            epoch_val_noisy_klnoise_loss = np.zeros((epochs,))
            epoch_val_noisy_dismiu_speech_loss = np.zeros((epochs,))
            epoch_val_noisy_dismiu_noise_loss = np.zeros((epochs,))
            epoch_val_noisy_residual_loss = np.zeros((epochs,))
            epoch_val_noisy_residual_loss_speech = np.zeros((epochs,))
            epoch_val_noisy_residual_loss_noise = np.zeros((epochs,))

            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = epochs
            best_clean_encoder_state_dict = clean_model_encoder.state_dict()
            best_clean_decoder_state_dict = clean_model_decoder.state_dict()
            best_noise_encoder_state_dict = noise_model_encoder.state_dict()
            best_noise_decoder_state_dict = noise_model_decoder.state_dict()
            best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()
        #   best_noisy_decoder_state_dict = noisy_model_decoder.state_dict()

            if flag_clean_encoder:
                    best_clean_encoder_optim_dict = optimizer_clean_en.state_dict()
                    best_clean_encoder_scheduler_dict = scheduler_clean_en.state_dict()
            if flag_clean_decoder:
                    best_clean_decoder_optim_dict = optimizer_clean_de.state_dict()
                    best_clean_decoder_scheduler_dict = scheduler_clean_de.state_dict()
            if flag_noise_encoder:
                    best_noise_encoder_optim_dict = optimizer_noise_en.state_dict()
                    best_noise_encoder_scheduler_dict = scheduler_noise_en.state_dict()
            if flag_noise_decoder:
                    best_noise_decoder_optim_dict = optimizer_noise_de.state_dict()
                    best_noise_decoder_scheduler_dict = scheduler_noise_de.state_dict()

            best_noisy_encoder_optim_dict = optimizer_noisy_en.state_dict()
            best_noisy_encoder_scheduler_dict = scheduler_noisy_en.state_dict()
        #   best_noisy_decoder_optim_dict = optimizer_noisy_de.state_dict()
            start_epoch = -1
    else:
            # resume training from certain epoch
            # load the model
            cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(model_name))
            checkpoint = torch.load(cp_file)
            clean_model_encoder.load_state_dict(checkpoint['clean_encoder_state_dict'])
            clean_model_decoder.load_state_dict(checkpoint['clean_decoder_state_dict'])
            noise_model_encoder.load_state_dict(checkpoint['noise_encoder_state_dict'])
            noise_model_decoder.load_state_dict(checkpoint['noise_decoder_state_dict'])
            noisy_model_encoder.load_state_dict(checkpoint['noisy_encoder_state_dict'])
        #   noisy_model_decoder.load_state_dict(checkpoint['noisy_decoder_state_dict'])

            if flag_clean_encoder:
                    optimizer_clean_en.load_state_dict(checkpoint['clean_encoder_optim_dict'])
                    scheduler_clean_en.load_state_dict(checkpoint['clean_encoder_scheduler_dict'])
            if flag_clean_decoder:
                    optimizer_clean_de.load_state_dict(checkpoint['clean_decoder_optim_dict'])
                    scheduler_clean_de.load_state_dict(checkpoint['clean_decoder_scheduler_dict'])
            if flag_noise_encoder:
                    optimizer_noise_en.load_state_dict(checkpoint['noise_encoder_optim_dict'])
                    scheduler_noise_en.load_state_dict(checkpoint['noise_encoder_scheduler_dict'])
            if flag_noise_decoder:
                    optimizer_noise_de.load_state_dict(checkpoint['noise_decoder_optim_dict'])
                    scheduler_noise_de.load_state_dict(checkpoint['noise_decoder_scheduler_dict'])
            optimizer_noisy_en.load_state_dict(checkpoint['noisy_encoder_optim_dict'])
            scheduler_noisy_en.load_state_dict(checkpoint['noisy_encoder_scheduler_dict'])
        #   optimizer_noisy_de.load_state_dict(checkpoint['noisy_decoder_optim_dict'])

            start_epoch = checkpoint['epoch']
            loss_log = checkpoint['loss_log']

            # load the loss
            epoch_train_total_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_total_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        #   epoch_train_noisy_recon_loss = np.pad(loss_log['train_noisy_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_kl_loss = np.pad(loss_log['train_noisy_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_klclean_loss = np.pad(loss_log['train_noisy_klclean'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_klnoise_loss = np.pad(loss_log['train_noisy_klnoise'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_dismiu_speech_loss = np.pad(loss_log['train_noisy_dismiu_speech'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_dismiu_noise_loss = np.pad(loss_log['train_noisy_dismiu_noise'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_residual_loss = np.pad(loss_log['train_noisy_residual'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_residual_loss_speech = np.pad(loss_log['train_noisy_residual_speech'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_train_noisy_residual_loss_noise = np.pad(loss_log['train_noisy_residual_noise'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        #   epoch_val_noisy_recon_loss = np.pad(loss_log['val_noisy_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_kl_loss = np.pad(loss_log['val_noisy_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_klclean_loss = np.pad(loss_log['val_noisy_klclean'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_klnoise_loss = np.pad(loss_log['val_noisy_klnoise'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_dismiu_speech_loss = np.pad(loss_log['train_noisy_dismiu_speech'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_dismiu_noise_loss = np.pad(loss_log['train_noisy_dismiu_noise'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_residual_loss = np.pad(loss_log['val_noisy_residual'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_residual_loss_speech = np.pad(loss_log['val_noisy_residual_speech'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            epoch_val_noisy_residual_loss_noise = np.pad(loss_log['val_noisy_residual_noise'], (0, epochs-start_epoch), mode='constant', constant_values=0)

            # load the current best model
            best_val_loss = checkpoint['best_val_loss']
            cpt_patience = checkpoint['cpt_patience']
            cur_best_epoch = start_epoch
            best_clean_encoder_state_dict = clean_model_encoder.state_dict()
            best_clean_decoder_state_dict = clean_model_decoder.state_dict()
            best_noise_encoder_state_dict = noise_model_encoder.state_dict()
            best_noise_decoder_state_dict = noise_model_decoder.state_dict()
            best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()
            # best_noisy_decoder_state_dict = noisy_model_decoder.state_dict()

            if flag_clean_encoder:
                    best_clean_encoder_optim_dict = optimizer_clean_en.state_dict()
                    best_clean_encoder_scheduler_dict = scheduler_clean_en.state_dict()
            if flag_clean_decoder:
                    best_clean_decoder_optim_dict = optimizer_clean_de.state_dict()
                    best_clean_decoder_scheduler_dict = scheduler_clean_de.state_dict()
            if flag_noise_encoder:
                    best_noise_encoder_optim_dict = optimizer_noise_en.state_dict()
                    best_noise_encoder_scheduler_dict = scheduler_noise_en.state_dict()
            if flag_noise_decoder:
                    best_noise_decoder_optim_dict = optimizer_noise_de.state_dict()
                    best_noise_decoder_scheduler_dict = scheduler_noise_de.state_dict()

            best_noisy_encoder_optim_dict = optimizer_noisy_en.state_dict()
            best_noisy_encoder_scheduler_dict = scheduler_noisy_en.state_dict()
            logger.info('Resuming trainning: epoch: {}'.format(start_epoch))            




    #################################################################################################################################       
    # npt_logger = NeptuneLogger(
    #        run=run,
    #        model=noisy_model_encoder,
    #        log_freq=30
    # )       

    for epoch in range(start_epoch+1, epochs):
        # loss
        train_total_loss = 0

    #   train_noisy_recon_loss = 0
        train_noisy_kl_loss = 0
        train_noisy_klclean_loss = 0
        train_noisy_klnoise_loss = 0
        train_noisy_dismiu_speech_loss = 0
        train_noisy_dismiu_noise_loss = 0
        train_noisy_residual_loss = 0
        train_noisy_residual_loss_speech = 0
        train_noisy_residual_loss_noise = 0

        val_total_loss = 0
    #   val_noisy_recon_loss = 0
        val_noisy_kl_loss = 0
        val_noisy_klclean_loss = 0
        val_noisy_klnoise_loss = 0
        val_noisy_dismiu_speech_loss = 0
        val_noisy_dismiu_noise_loss = 0
        val_noisy_residual_loss = 0
        val_noisy_residual_loss_speech = 0
        val_noisy_residual_loss_noise = 0

        # training
        if flag_noise_encoder:
                noise_model_encoder.train()
        else:
                noise_model_encoder.eval()
        if flag_noise_decoder:
                noise_model_decoder.train()
        else:
                noise_model_decoder.eval()              
        if flag_clean_encoder:
                clean_model_encoder.train()
        else:
                clean_model_encoder.eval()
        if flag_clean_decoder:
                clean_model_decoder.train()
        else:
                clean_model_decoder.eval()

        noisy_model_encoder.train()
    #   noisy_model_decoder.train()

        start_time = datetime.datetime.now()
        for i, batch_data in enumerate(train_dataloader):

            noisy_batch, clean_batch, noise_batch = batch_data[0], batch_data[1], batch_data[2]
            noisy_batch = noisy_batch.to(device) # [batch, time, freq]
            clean_batch = clean_batch.to(device)
            noise_batch = noise_batch.to(device)

            bs, time_len = noisy_batch.shape

            noisy_batch = noisy_batch.float()
            clean_batch = clean_batch.float()
            noise_batch = noise_batch.float()

            # clean encoder
            if flag_clean_encoder:
                z_clean, miu_clean, log_sigma_clean, delta_clean, skiper_clean, C_clean, F_clean, stft_x_clean = clean_model_encoder(clean_batch, train=True)
            else:
                with torch.no_grad():
                        z_clean, miu_clean, log_sigma_clean, delta_clean, skiper_clean, C_clean, F_clean, stft_x_clean = clean_model_encoder(clean_batch, train=False)
            # check_and_log_nan(miu_clean, 'miu_clean')
            # check_and_log_nan(log_sigma_clean, 'log_sigma_clean')
            # check_and_log_nan(delta_clean,'delta_clean')
            # noise encoder
            if flag_noise_encoder:           
                z_noise, miu_noise, log_sigma_noise, delta_noise, skiper_noise, C_noise, F_noise, stft_x_noise = noise_model_encoder(noise_batch, train=True)
            else:
                with torch.no_grad():
                    z_noise, miu_noise, log_sigma_noise, delta_noise, skiper_noise, C_noise, F_noise, stft_x_noise = noise_model_encoder(noise_batch, train=False)
            # check_and_log_nan(miu_noise, 'miu_noise')
            # check_and_log_nan(log_sigma_noise, 'log_sigma_noise')
            # check_and_log_nan(delta_noise, 'delta_noise')
            # noisy encoder
            # z_noisy, miu_noisy, log_sigma_noisy, delta_noisy, skiper_noisy, C_noisy, F_noisy, stft_x_noisy = noisy_model_encoder(noisy_batch, train=True)
            (z_speech_noisy, miu_speech_noisy, log_sigma_speech_noisy, delta_speech_noisy, 
             z_noise_noisy, miu_noise_noisy, log_sigma_noise_noisy, delta_noise_noisy, 
             skiper_noisy, C_noisy, F_noisy, stft_x_noisy) = noisy_model_encoder(noisy_batch, train=True)
            
            # check_and_log_nan(z_speech_noisy, 'z_speech_noisy')
            # check_and_log_nan(miu_speech_noisy, 'miu_speech_noisy')
            # check_and_log_nan(log_sigma_speech_noisy, 'log_sigma_speech_noisy')
            # check_and_log_nan(delta_speech_noisy, 'delta_speech_noisy')
            # check_and_log_nan(z_noise_noisy, 'z_noise_noisy')
            # check_and_log_nan(miu_noise_noisy, 'miu_noise_noisy')
            # check_and_log_nan(log_sigma_noise_noisy, 'log_sigma_noise_noisy')
            # check_and_log_nan(delta_noise_noisy, 'delta_noise_noisy')
            # check_and_log_nan(delta_noise)
            # print(z_clean[24, 34,:].shape)
            # print(z_clean[24, 34,:])
            # check_and_log_nan(z_noisy, 'z_noisy')
            # noisy decoder
        #  noisy_mean, noisy_log_var, es_noisy = noisy_model_decoder(z_noisy_clean, z_noisy_noise)

            ####### whether train cvae decoder and nvae decoder together#######################
            # # clean decoder
            # if flag_clean_decoder:
                # recon_sig, _ = clean_model_decoder(stft_x_noisy, z_clean, skiper_clean, C_clean, F_clean, train=False)
            # else:
            #     with torch.no_grad():
            #             recon_sig, _, _ = clean_model_decoder(stft_x, z, skiper, C, F, train=True)
            # # noise decoder
            # if flag_noise_decoder:
            #     noise_mean, noise_log_var, es_noise = noise_model_decoder(z_noise)
            # else:
            #     with torch.no_grad():
            #             noise_mean, noise_log_var, es_noise = noise_model_decoder(z_noise)

            # calculate loss
            total_loss, kl_loss, kl_clean, kl_noise, dismiu_speech, dismiu_noise, residual_loss, resi_speech, resi_noise = \
            model_loss.final_nsvae_loss(miu_clean, miu_noise, miu_speech_noisy, miu_noise_noisy,
                                         log_sigma_clean, log_sigma_noise, log_sigma_speech_noisy, log_sigma_noise_noisy, 
                                        delta_clean, delta_noise, delta_speech_noisy, delta_noise_noisy,
                                        z_speech_noisy, z_noise_noisy,
                                        skiper_clean, skiper_noise, skiper_noisy)
            # print(kl_clean,kl_noise, resi_speech)   
            # check_and_log_nan(kl_clean, 'kl_clean')
            # check_and_log_nan(kl_noise, 'kl_noise') 

    

            # backward
            if flag_clean_encoder:
                optimizer_clean_en.zero_grad()
            if flag_clean_decoder:
                optimizer_clean_de.zero_grad()
            if flag_noise_encoder:
                optimizer_noise_en.zero_grad()
            if flag_noise_decoder:
                optimizer_noise_de.zero_grad()
            optimizer_noisy_en.zero_grad()
        #  optimizer_noisy_de.zero_grad()  

            total_loss.backward()

        #  optimizer_noisy_de.step()
            optimizer_noisy_en.step()
            if flag_clean_encoder:
                optimizer_clean_en.step()
            if flag_clean_decoder:
                optimizer_clean_de.step()
            if flag_noise_encoder:
                optimizer_noise_en.step()
            if flag_noise_decoder:
                optimizer_noise_de.step()

                    


            # loss per time frame
            train_total_loss += total_loss * bs
        #  train_noisy_recon_loss += noisy_restore_loss
            train_noisy_kl_loss += kl_loss * bs 
            train_noisy_klclean_loss += kl_clean * bs
            train_noisy_klnoise_loss += kl_noise * bs
            train_noisy_dismiu_speech_loss += dismiu_speech * bs
            train_noisy_dismiu_noise_loss += dismiu_noise * bs
            train_noisy_residual_loss += residual_loss * bs
            train_noisy_residual_loss_speech += resi_speech * bs
            train_noisy_residual_loss_noise += resi_noise * bs

        # loss per sample per time frame
        epoch_train_total_loss[epoch] = train_total_loss / train_num
    #   epoch_train_noisy_recon_loss[epoch] = train_noisy_recon_loss * input_dim / train_num
        epoch_train_noisy_kl_loss[epoch] = train_noisy_kl_loss / train_num
        epoch_train_noisy_klclean_loss[epoch] = train_noisy_klclean_loss / train_num
        epoch_train_noisy_klnoise_loss[epoch] = train_noisy_klnoise_loss / train_num
        epoch_train_noisy_dismiu_speech_loss[epoch] = train_noisy_dismiu_speech_loss / train_num
        epoch_train_noisy_dismiu_noise_loss[epoch] = train_noisy_dismiu_noise_loss / train_num
        epoch_train_noisy_residual_loss[epoch] = train_noisy_residual_loss / train_num      
        epoch_train_noisy_residual_loss_speech[epoch] = train_noisy_residual_loss_speech / train_num
        epoch_train_noisy_residual_loss_noise[epoch] = train_noisy_residual_loss_noise / train_num      
       


        # validation
        clean_model_encoder.eval()
        clean_model_decoder.eval()
        noise_model_encoder.eval()
        noise_model_decoder.eval()
        noisy_model_encoder.eval()
        #   noisy_model_decoder.eval()
        for i, batch_data in enumerate(val_dataloader):

            noisy_batch, clean_batch, noise_batch = batch_data[0], batch_data[1], batch_data[2]
            noisy_batch = noisy_batch.to(device) # [batch, time, freq]
            clean_batch = clean_batch.to(device)
            noise_batch = noise_batch.to(device)

            bs, time_len = noisy_batch.shape

            noisy_batch = noisy_batch.float()
            clean_batch = clean_batch.float()
            noise_batch = noise_batch.float()

            # clean encoder
            with torch.no_grad():
                z_clean, miu_clean, log_sigma_clean, delta_clean, skiper_clean, C_clean, F_clean, stft_x_clean = clean_model_encoder(clean_batch, train=False)
                z_noise, miu_noise, log_sigma_noise, delta_noise, skiper_noise, C_noise, F_noise, stft_x_noise = noise_model_encoder(noise_batch, train=False)
                (z_speech_noisy, miu_speech_noisy, log_sigma_speech_noisy, delta_speech_noisy, 
                    z_noise_noisy, miu_noise_noisy, log_sigma_noise_noisy, delta_noise_noisy, 
                    skiper_noisy, C_noisy, F_noisy, stft_x_noisy) = noisy_model_encoder(noisy_batch, train=False)            # # noisy decoder
            # # noisy_mean, noisy_log_var, es_noisy = noisy_model_decoder(z_noisy_clean, z_noisy_noise)
            # # clean decoder
            # clean_mean, clean_log_var, es_clean = clean_model_decoder(z_clean)
            # # noise decoder
            # noise_mean, noise_log_var, es_noise = noise_model_decoder(z_noise)

            # calculate loss
            total_loss, kl_loss, kl_clean, kl_noise, dismiu_speech, dismiu_noise, residual_loss, resi_speech, resi_noise = \
            model_loss.final_nsvae_loss(miu_clean, miu_noise, miu_speech_noisy, miu_noise_noisy,
                                         log_sigma_clean, log_sigma_noise, log_sigma_speech_noisy, log_sigma_noise_noisy, 
                                        delta_clean, delta_noise, delta_speech_noisy, delta_noise_noisy,
                                        z_speech_noisy, z_noise_noisy,
                                        skiper_clean, skiper_noise, skiper_noisy)


            # loss
            val_total_loss += total_loss * bs
        #  val_noisy_recon_loss += noisy_restore_loss
            val_noisy_kl_loss += kl_loss * bs
            val_noisy_klclean_loss += kl_clean * bs
            val_noisy_klnoise_loss += kl_noise * bs
            val_noisy_dismiu_speech_loss += dismiu_speech * bs
            val_noisy_dismiu_noise_loss += dismiu_noise * bs
            val_noisy_residual_loss += residual_loss * bs
            val_noisy_residual_loss_speech += resi_speech * bs
            val_noisy_residual_loss_noise += resi_noise * bs
            
        # loss per sample per time frame
        epoch_val_total_loss[epoch] = val_total_loss / val_num
    #   epoch_val_noisy_recon_loss[epoch] = val_noisy_recon_loss * input_dim / val_num
        epoch_val_noisy_kl_loss[epoch] = val_noisy_kl_loss / val_num
        epoch_val_noisy_klclean_loss[epoch] = val_noisy_klclean_loss / val_num
        epoch_val_noisy_klnoise_loss[epoch] = val_noisy_klnoise_loss / val_num  
        epoch_val_noisy_dismiu_speech_loss[epoch] = val_noisy_dismiu_speech_loss / val_num
        epoch_val_noisy_dismiu_noise_loss[epoch] = val_noisy_dismiu_noise_loss / val_num   
        epoch_val_noisy_residual_loss[epoch] = val_noisy_residual_loss / val_num 
        epoch_val_noisy_residual_loss_speech[epoch] = val_noisy_residual_loss_speech / val_num
        epoch_val_noisy_residual_loss_noise[epoch] = val_noisy_residual_loss_noise / val_num
 
        scheduler_noisy_en.step(epoch_val_total_loss[epoch])
        if flag_clean_encoder:
              scheduler_clean_en.step(epoch_val_total_loss[epoch])
        if flag_clean_decoder:
              scheduler_clean_de.step(epoch_val_total_loss[epoch])
        if flag_noise_encoder:
              scheduler_noise_en.step(epoch_val_total_loss[epoch])
        if flag_noise_decoder:
              scheduler_noise_de.step(epoch_val_total_loss[epoch])


        # save the current best model according to total val loss
        if epoch_val_total_loss[epoch] < best_val_loss:
            best_val_loss = epoch_val_total_loss[epoch]
            cpt_patience = 0
            best_clean_encoder_state_dict = clean_model_encoder.state_dict()
            best_clean_decoder_state_dict = clean_model_decoder.state_dict()
            best_noise_encoder_state_dict = noise_model_encoder.state_dict()
            best_noise_decoder_state_dict = noise_model_decoder.state_dict()
            best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()

            if flag_clean_encoder:
                best_clean_encoder_optim_dict = optimizer_clean_en.state_dict()
                best_clean_encoder_scheduler_dict = scheduler_clean_en.state_dict()
            if flag_clean_decoder:
                best_clean_decoder_optim_dict = optimizer_clean_de.state_dict()
                best_clean_decoder_scheduler_dict = scheduler_clean_de.state_dict()
            if flag_noise_encoder:
                best_noise_encoder_optim_dict = optimizer_noise_en.state_dict()
                best_noise_encoder_scheduler_dict = scheduler_noise_en.state_dict()
            if flag_noise_decoder:
                best_noise_decoder_optim_dict = optimizer_noise_de.state_dict()
                best_noise_decoder_scheduler_dict = scheduler_noise_de.state_dict()

            best_noisy_encoder_optim_dict = optimizer_noisy_en.state_dict()
            best_noisy_encoder_scheduler_dict = scheduler_noisy_en.state_dict()
            cur_best_epoch = epoch

            save_file = os.path.join(save_dir, model_name + '_clean_encoder_best_epoch.pt')
            torch.save(best_clean_encoder_state_dict, save_file)

            save_file = os.path.join(save_dir, model_name + '_clean_decoder_best_epoch.pt')
            torch.save(best_clean_decoder_state_dict, save_file)

            save_file = os.path.join(save_dir, model_name + '_noise_encoder_best_epoch.pt')
            torch.save(best_noise_encoder_state_dict, save_file)

            save_file = os.path.join(save_dir, model_name + '_noise_decoder_best_epoch.pt')
            torch.save(best_noise_decoder_state_dict, save_file)

            save_file = os.path.join(save_dir, model_name + '_noisy_encoder_best_epoch.pt')
            torch.save(best_noisy_encoder_state_dict, save_file)

            loss_log = {'train_loss': epoch_train_total_loss[:cur_best_epoch+1],
                        'val_loss': epoch_val_total_loss[:cur_best_epoch+1],

                        'train_noisy_kl': epoch_train_noisy_kl_loss[:cur_best_epoch+1],
                        'train_noisy_klclean': epoch_train_noisy_klclean_loss[:cur_best_epoch+1],
                        'train_noisy_klnoise': epoch_train_noisy_klnoise_loss[:cur_best_epoch+1],
                        'train_noisy_dismiu_speech': epoch_train_noisy_dismiu_speech_loss[:cur_best_epoch+1],
                        'train_noisy_dismiu_noise': epoch_train_noisy_dismiu_noise_loss[:cur_best_epoch+1],
                        'train_noisy_residual': epoch_train_noisy_residual_loss[:cur_best_epoch+1],
                        'train_noisy_residual_speech': epoch_train_noisy_residual_loss_speech[:cur_best_epoch+1],
                        'train_noisy_residual_noise': epoch_train_noisy_residual_loss_noise[:cur_best_epoch+1],

                        'val_noisy_kl':epoch_val_noisy_kl_loss[:cur_best_epoch+1],
                        'val_noisy_klclean': epoch_val_noisy_klclean_loss[:cur_best_epoch+1],
                        'val_noisy_klnoise': epoch_val_noisy_klnoise_loss[:cur_best_epoch+1],
                        'val_noisy_dismiu_speech': epoch_val_noisy_dismiu_speech_loss[:cur_best_epoch+1],
                        'val_noisy_dismiu_noise': epoch_val_noisy_dismiu_noise_loss[:cur_best_epoch+1],
                        'val_noisy_residual': epoch_val_noisy_residual_loss[:cur_best_epoch+1],
                        'val_noisy_residual_speech': epoch_val_noisy_residual_loss_speech[:cur_best_epoch+1],
                        'val_noisy_residual_noise': epoch_val_noisy_residual_loss_noise[:cur_best_epoch+1],
                }

            save_file = os.path.join(save_dir, model_name + '_checkpoint.pt')
            save_dict = {'epoch': cur_best_epoch,
                        'best_val_loss': best_val_loss,
                        'cpt_patience': cpt_patience,
                        'clean_encoder_state_dict': best_clean_encoder_state_dict,
                        'clean_decoder_state_dict': best_clean_decoder_state_dict,
                        'noise_encoder_state_dict': best_noise_encoder_state_dict,
                        'noise_decoder_state_dict': best_noise_decoder_state_dict,
                        'noisy_encoder_state_dict': best_noisy_encoder_state_dict,
                    #    'noisy_decoder_state_dict': best_noisy_decoder_state_dict,
                        'loss_log': loss_log
                }
            if flag_clean_encoder:
                save_dict['clean_encoder_optim_dict'] = best_clean_encoder_optim_dict
                save_dict['clean_encoder_scheduler_dict'] = best_clean_encoder_scheduler_dict
            if flag_clean_decoder:
                save_dict['clean_decoder_optim_dict'] = best_clean_decoder_optim_dict
                save_dict['clean_decoder_scheduler_dict'] = best_clean_decoder_scheduler_dict
            if flag_noise_encoder:
                save_dict['noise_encoder_optim_dict'] = best_noise_encoder_optim_dict
                save_dict['noise_encoder_scheduler_dict'] = best_noise_encoder_scheduler_dict
            if flag_noise_decoder:
                save_dict['noise_decoder_optim_dict'] = best_noise_decoder_optim_dict
                save_dict['noise_decoder_scheduler_dict'] = best_noise_decoder_scheduler_dict
            
            save_dict['noisy_encoder_optim_dict'] = best_noisy_encoder_optim_dict
            save_dict['noisy_encoder_scheduler_dict'] = best_noisy_encoder_scheduler_dict
        #  save_dict['noisy_decoder_optim_dict'] = best_noisy_decoder_optim_dict
            torch.save(save_dict, save_file)
    
            logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))    
        else:
            cpt_patience += 1   

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
        logger.info('Train => tot: {:.2f} clean kl {:.2f} noise kl {:.2f} miudis speech {:.4f} miudis noise {:.4f} noisy kl {:.2f} residual {:.2f} resi_speech {:.2f} resi_noise {:.2f}'.format(epoch_train_total_loss[epoch], 
                                                                                                                    epoch_train_noisy_klclean_loss[epoch], epoch_train_noisy_klnoise_loss[epoch], 
                                                                                                                    epoch_train_noisy_dismiu_speech_loss[epoch], epoch_train_noisy_dismiu_noise_loss[epoch],
                                                                                                                    epoch_train_noisy_kl_loss[epoch],
                                                                                                                    epoch_train_noisy_residual_loss[epoch],epoch_train_noisy_residual_loss_speech[epoch],
                                                                                                                    epoch_train_noisy_residual_loss_noise[epoch]))
        logger.info('Val => tot: {:.2f} clean kl {:.2f} noise kl {:.2f} miudis speech {:.4f} miudis noise {:.4f} noisy kl {:.2f} residual {:.2f} resi_speech {:.2f} resi_noise {:.2f}'.format(epoch_val_total_loss[epoch], 
                                                                                                                     epoch_val_noisy_klclean_loss[epoch],  epoch_val_noisy_klnoise_loss[epoch], 
                                                                                                                     epoch_val_noisy_dismiu_speech_loss[epoch], epoch_val_noisy_dismiu_noise_loss[epoch],
                                                                                                                     epoch_val_noisy_kl_loss[epoch],
                                                                                                                    epoch_val_noisy_residual_loss[epoch],epoch_val_noisy_residual_loss_speech[epoch],
                                                                                                                    epoch_val_noisy_residual_loss_noise[epoch]))
        # Stop traning if early-stop triggers
        if cpt_patience == early_stop_patience:
            logger.info('Early stop patience achieved')
            break 


    # Save the training loss and validation loss
    train_loss_log = epoch_train_total_loss[:epoch+1]
    #    train_noisy_recon_log = epoch_train_noisy_recon_loss[:epoch+1]
    train_noisy_kl_log = epoch_train_noisy_kl_loss[:epoch+1]
    train_noisy_klclean_log = epoch_train_noisy_klclean_loss[:epoch+1]
    train_noisy_klnoise_log = epoch_train_noisy_klnoise_loss[:epoch+1]
    train_noisy_dismiu_speech_log = epoch_train_noisy_dismiu_speech_loss[:epoch+1]
    train_noisy_dismiu_noise_log = epoch_train_noisy_dismiu_noise_loss[:epoch+1]
    train_noisy_residual_log = epoch_train_noisy_residual_loss[:epoch+1]
    train_noisy_residual_log_speech = epoch_train_noisy_residual_loss_speech[:epoch+1]
    train_noisy_residual_log_noise = epoch_train_noisy_residual_loss_noise[:epoch+1]


    val_loss_log = epoch_val_total_loss[:epoch+1]
    #    val_noisy_recon_log = epoch_val_noisy_recon_loss[:epoch+1]
    val_noisy_kl_log = epoch_val_noisy_kl_loss[:epoch+1]
    val_noisy_klclean_log = epoch_val_noisy_klclean_loss[:epoch+1]
    val_noisy_klnoise_log = epoch_val_noisy_klnoise_loss[:epoch+1]
    val_noisy_dismiu_speech_log = epoch_val_noisy_dismiu_speech_loss[:epoch+1]
    val_noisy_dismiu_noise_log = epoch_val_noisy_dismiu_noise_loss[:epoch+1]
    val_noisy_residual_log = epoch_val_noisy_residual_loss[:epoch+1]
    val_noisy_residual_log_speech = epoch_val_noisy_residual_loss_speech[:epoch+1]
    val_noisy_residual_log_noise = epoch_val_noisy_residual_loss_noise[:epoch+1]

    loss_file = os.path.join(save_dir, 'loss_model.pckl')
    with open(loss_file, 'wb') as f:
            save_list = [train_loss_log, 
                        train_noisy_kl_log,
                        train_noisy_klclean_log, train_noisy_klnoise_log,
                        train_noisy_dismiu_speech_log, train_noisy_dismiu_noise_log ,train_noisy_residual_log,
                        train_noisy_residual_log_speech, train_noisy_residual_log_noise,
                        val_loss_log,
                        val_noisy_kl_log, 
                        val_noisy_klclean_log, val_noisy_klnoise_log,
                        val_noisy_dismiu_speech_log, val_noisy_dismiu_noise_log, val_noisy_residual_log,
                        val_noisy_residual_log_speech, val_noisy_residual_log_noise]
            pickle.dump(save_list, f)









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='cfg file for training')
    parser.add_argument('--reload', action='store_true', help='whether to load existed model')
    parser.add_argument('--model_dir', type=str, default='None', help='if reload existed model, what is the model dir')
    parser.add_argument('--first_use_dataset', action='store_true', help='first use this dataset?')
    parser.add_argument('--causal', action='store_true', help='whether use causal verdion')
    parser.add_argument('--nsvae_model', type=str, default='original', help='what nsvae structure to use')
    parser.add_argument('--latent_num', type=int, default=2, help='num of latent vectors from nsvae encoder')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of noise kl')
    parser.add_argument('--w_resi', type=float, default=0.0, help='weight of the residual loss')
    parser.add_argument('--matching', type=str, default='speech', help='what skipc to match, noise or speech or both')
    parser.add_argument('--w_kl', type=float, default=1.0, help='weight of the kl loss')
    parser.add_argument('--w_dismiu', type=float, default=0)
    parser.add_argument('--num_samples', type=int, default=1, help='num of z sampled')
    parser.add_argument('--zdim', type=int, default=128, help='dimension of z')

    args = parser.parse_args()
    cfg = myconf()
    cfg.read(args.cfg_file)

    log_params = {
        'reload': args.reload,
        'cfg_file': args.cfg_file,
        'model_dir': args.model_dir,
        'first_use_dataset': args.first_use_dataset,
        'causal': args.causal,
        'nsvae_model': args.nsvae_model,
        'latent_num': args.latent_num,
        'alpha': args.alpha,
        'w_resi': args.w_resi,
        'matching': args.matching,
        'w_kl': args.w_kl,
        'w_dismiu': args.w_dismiu,
        'num_samples': args.num_samples,
        'zdim': args.zdim

    }



    beta_pvae(cfg, log_params)

