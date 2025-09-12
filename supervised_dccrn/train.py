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

main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_folder_path)

from model.pvae_module import DCCRN_
from dataset import dataload_supervised_dccrn
import datetime
import argparse
import socket
import os
import shutil
from utils.logger import get_logger
import pickle
from utils.read_config import myconf
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig
from model.nsvae_loss import ete_train_se_loss

import psutil
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_memory(step):
    process = psutil.Process(os.getpid())
    print(f"[{step}] CPU Memory: {process.memory_info().rss / 1e6:.2f} MB")
    if torch.cuda.is_available():
        print(f"[{step}] GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"[{step}] GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")


def supervised_dccrn(cfg, log_params):

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
    save_frequency = cfg.getint('Training', 'save_frequency')
    learning_rate = cfg.getfloat('Training', 'lr')


    # load model
    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft') 
    causal = log_params['causal']
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()
    tmp_skip_to_use = log_params['skip_to_use']
    skip_to_use = []
    for i in tmp_skip_to_use:
        skip_to_use.append(int(i))
    print(skip_to_use)  
    recon_type=log_params['recon_type']
    resyn = log_params['resynthesis']
    if log_params['data_norm']:
        data_mean = log_params['data_mean']
        data_mean = torch.tensor(data_mean, dtype=torch.float32).to(device)
        data_std = log_params['data_std']
        data_std = torch.tensor(data_std, dtype=torch.float32).to(device)
    else:
        data_mean = None
        data_std = None        
    model = DCCRN_(nfft, hop, net_params, causal, device, wlen, skip_to_use, recon_type, resyn, data_mean, data_std)
    model.to(device)

    # define loss
    tmp_recon_w = log_params['recon_loss_weight']
    recon_weight = []
    for w in tmp_recon_w:
        w = float(w)
        recon_weight.append(w)
    print(recon_weight)
    model_loss = ete_train_se_loss(recon_weight)
    # model_loss.to(device)

    optimizer_model = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, 'min', factor=0.5,patience=3)
    
    if not log_params['reload']:
        saved_root = cfg.get('User', 'saved_root')
        filename = "{}_{}_causal={}_skipuse={}_reconw={}_recontype={}_resynthesis={}_datanorm={}".format(date, model_name, causal, tmp_skip_to_use, tmp_recon_w, recon_type, resyn, log_params['data_norm'])
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
                os.makedirs(save_dir)
    else:
    #     tag = self.cfg.get('Network', 'tag')
        save_dir = log_params['model_dir']

    # save the model configuration
    save_cfg = os.path.join(save_dir, 'config.ini')
    shutil.copy(log_params['cfg_file'], save_cfg)
    



    # create logger
    log_file = os.path.join(save_dir, 'log.txt')
    logger_type = cfg.getint('User', 'logger_type')
    logger = get_logger(log_file, logger_type)

    # Print basical infomation
    for log in basic_info:
            logger.info(log)
    logger.info('In this experiment, result will be saved in: ' + save_dir) 
    
    
    
    # load data
    first_use_dataset = log_params['first_use_dataset']
    train_dataloader, val_dataloader, train_num, val_num = dataload_supervised_dccrn.build_dataloader(cfg, first_use_dataset)
    logger.info('Train on {}'.format(dataset_name))
    logger.info('Training samples: {}'.format(train_num))
    logger.info('Validation samples: {}'.format(val_num))  

    # Create python list for loss
    if not log_params['reload']:
        #train loss
        epoch_train_total_loss = np.zeros((epochs,))
        epoch_train_noisy_recon_loss_cpx = np.zeros((epochs,))
        epoch_train_noisy_recon_loss_mag = np.zeros((epochs,))
        epoch_train_noisy_recon_loss_sisnr = np.zeros((epochs,))
        # val loss
        epoch_val_total_loss = np.zeros((epochs,))
        epoch_val_noisy_recon_loss_cpx = np.zeros((epochs,))
        epoch_val_noisy_recon_loss_mag = np.zeros((epochs,))
        epoch_val_noisy_recon_loss_sisnr = np.zeros((epochs,))

        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_model_state_dict = model.state_dict()

        best_model_optim_dict = optimizer_model.state_dict()
        best_model_scheduler_dict = scheduler.state_dict()
        start_epoch = -1
    else:
        # resume training from certain epoch
        # load the model
        cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(model_name))
        checkpoint = torch.load(cp_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_model.load_state_dict(checkpoint['model_optim_dict'])
        scheduler.load_state_dict(checkpoint['model_scheduler_dict'])

        start_epoch = checkpoint['epoch']
        loss_log = checkpoint['loss_log']

        # load the loss
        epoch_train_total_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_total_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_noisy_recon_loss_cpx = np.pad(loss_log['train_noisy_recon_cpx'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_noisy_recon_loss_mag = np.pad(loss_log['train_noisy_recon_mag'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_noisy_recon_loss_sisnr = np.pad(loss_log['train_noisy_recon_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_noisy_recon_loss_cpx = np.pad(loss_log['val_noisy_recon_cpx'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_noisy_recon_loss_mag = np.pad(loss_log['val_noisy_recon_mag'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_noisy_recon_loss_sisnr = np.pad(loss_log['val_noisy_recon_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        # load the current best model
        best_val_loss = checkpoint['best_val_loss']
        cpt_patience = checkpoint['cpt_patience']
        cur_best_epoch = start_epoch
        best_model_state_dict = model.state_dict()

        best_model_optim_dict = optimizer_model.state_dict()
        best_model_scheduler_dict = scheduler.state_dict()

        logger.info('Resuming trainning: epoch: {}'.format(start_epoch))            
    
    
    

    for epoch in range(start_epoch+1, epochs):
        # loss
        train_total_loss = 0
        train_noisy_recon_loss_cpx = 0
        train_noisy_recon_loss_mag = 0
        train_noisy_recon_loss_sisnr = 0

        val_total_loss = 0
        val_noisy_recon_loss_cpx = 0
        val_noisy_recon_loss_mag = 0
        val_noisy_recon_loss_sisnr = 0

        # training
        model.train()

        start_time = datetime.datetime.now()
        for i, batch_data in enumerate(train_dataloader):

            noisy_batch, clean_batch = batch_data[0], batch_data[1]
            noisy_batch = noisy_batch.to(device) # [batch, time]
            bs = noisy_batch.shape[0]
            clean_batch = clean_batch.to(device)


            noisy_batch = noisy_batch.float()
            clean_batch = clean_batch.float()


            estimated_clean, est_clean_stft = model(noisy_batch)
            clean_stft = model.stft(clean_batch)

            # calculate loss
            total_loss, loss_cpx, loss_mag, sisnr_loss = model_loss.final_ete_loss(est_clean_stft, clean_stft, clean_batch, estimated_clean)
            
            optimizer_model.zero_grad()

            total_loss.backward()

            optimizer_model.step()

            # loss per input audio
            train_total_loss += total_loss * bs
            train_noisy_recon_loss_cpx += loss_cpx * bs
            train_noisy_recon_loss_mag += loss_mag * bs
            train_noisy_recon_loss_sisnr += sisnr_loss * bs

        epoch_train_total_loss[epoch] = train_total_loss / train_num 
        epoch_train_noisy_recon_loss_cpx[epoch] = train_noisy_recon_loss_cpx / train_num
        epoch_train_noisy_recon_loss_mag[epoch] = train_noisy_recon_loss_mag / train_num      
        epoch_train_noisy_recon_loss_sisnr[epoch] = train_noisy_recon_loss_sisnr / train_num 

     


        # validation
        model.eval()
        for i, batch_data in enumerate(val_dataloader):

            noisy_batch, clean_batch = batch_data[0], batch_data[1]
            noisy_batch = noisy_batch.to(device) # [batch, time]
            bs = noisy_batch.shape[0]
            clean_batch = clean_batch.to(device)


            noisy_batch = noisy_batch.float()
            clean_batch = clean_batch.float()

            with torch.no_grad():
                estimated_clean, est_clean_stft = model(noisy_batch, train=False)
                clean_stft = model.stft(clean_batch)

            total_loss, loss_cpx, loss_mag, sisnr_loss = model_loss.final_ete_loss(est_clean_stft, clean_stft, clean_batch, estimated_clean)


            # loss
            val_total_loss += total_loss * bs
            val_noisy_recon_loss_cpx += loss_cpx * bs
            val_noisy_recon_loss_mag += loss_mag * bs
            val_noisy_recon_loss_sisnr += sisnr_loss * bs

        epoch_val_total_loss[epoch] = val_total_loss / val_num
        epoch_val_noisy_recon_loss_cpx[epoch] = val_noisy_recon_loss_cpx / val_num
        epoch_val_noisy_recon_loss_mag[epoch] = val_noisy_recon_loss_mag / val_num      
        epoch_val_noisy_recon_loss_sisnr[epoch] = val_noisy_recon_loss_sisnr / val_num  

        scheduler.step(epoch_val_total_loss[epoch]) 

    

        # save the current best model according to total val loss
        if epoch_val_total_loss[epoch] < best_val_loss:
            best_val_loss = epoch_val_total_loss[epoch]
            cpt_patience = 0
            best_model_state_dict = model.state_dict()
            best_model_optim_dict = optimizer_model.state_dict()
            best_model_scheduler_dict = scheduler.state_dict()
            cur_best_epoch = epoch
            save_file = os.path.join(save_dir, model_name + '_curr_best_epoch.pt')
            torch.save(best_model_state_dict, save_file)

            loss_log = {'train_loss': epoch_train_total_loss[:cur_best_epoch+1],
                        'val_loss': epoch_val_total_loss[:cur_best_epoch+1],
                        'train_noisy_recon_cpx': epoch_train_noisy_recon_loss_cpx[:cur_best_epoch+1],
                        'train_noisy_recon_mag': epoch_train_noisy_recon_loss_mag[:cur_best_epoch+1],
                        'train_noisy_recon_sisnr': epoch_train_noisy_recon_loss_sisnr[:cur_best_epoch+1],    
                        'val_noisy_recon_cpx': epoch_val_noisy_recon_loss_cpx[:cur_best_epoch+1],
                        'val_noisy_recon_mag': epoch_val_noisy_recon_loss_mag[:cur_best_epoch+1],
                        'val_noisy_recon_sisnr': epoch_val_noisy_recon_loss_sisnr[:cur_best_epoch+1]                    
                }

            save_file = os.path.join(save_dir, model_name + '_checkpoint.pt')
            save_dict = {'epoch': cur_best_epoch,
                        'best_val_loss': best_val_loss,
                        'cpt_patience': cpt_patience,
                        'model_state_dict': best_model_state_dict,
                        'loss_log': loss_log
                }
            save_dict['model_optim_dict'] = best_model_optim_dict
            save_dict['model_scheduler_dict'] = best_model_scheduler_dict
            torch.save(save_dict, save_file)
            
            logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))
        else:
            cpt_patience += 1   

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
        logger.info('Train => tot: {:.2f} recon_cpx {:.2f} recon mag {:.2f} recon sisnr {:.2f}'.format(epoch_train_total_loss[epoch], epoch_train_noisy_recon_loss_cpx[epoch],
                                                                                                        epoch_train_noisy_recon_loss_mag[epoch], epoch_train_noisy_recon_loss_sisnr[epoch]))  
        logger.info('Val => tot: {:.2f} recon_cpx {:.2f} recon mag {:.2f} recon sisnr {:.2f}'.format(epoch_val_total_loss[epoch], epoch_val_noisy_recon_loss_cpx[epoch],
                                                                                                        epoch_val_noisy_recon_loss_mag[epoch], epoch_val_noisy_recon_loss_sisnr[epoch])) 

  

        if cpt_patience == early_stop_patience:
            logger.info('Early stop patience achieved')
            break 



    
    # Save the training loss and validation loss
    train_loss_log = epoch_train_total_loss[:epoch+1]
    train_noisy_recon_log_cpx = epoch_train_noisy_recon_loss_cpx[:epoch+1]
    train_noisy_recon_log_mag = epoch_train_noisy_recon_loss_mag[:epoch+1]
    train_noisy_recon_log_sisnr = epoch_train_noisy_recon_loss_sisnr[:epoch+1]

    val_noisy_recon_log_cpx = epoch_val_noisy_recon_loss_cpx[:epoch+1]
    val_noisy_recon_log_mag = epoch_val_noisy_recon_loss_mag[:epoch+1]
    val_noisy_recon_log_sisnr = epoch_val_noisy_recon_loss_sisnr[:epoch+1]

    val_loss_log = epoch_val_total_loss[:epoch+1]
    
    loss_file = os.path.join(save_dir, 'loss_model.pckl')
    with open(loss_file, 'wb') as f:
        save_list = [train_loss_log, val_loss_log, train_noisy_recon_log_cpx, train_noisy_recon_log_mag, train_noisy_recon_log_sisnr,
                     val_noisy_recon_log_cpx, val_noisy_recon_log_mag, val_noisy_recon_log_sisnr]
        pickle.dump(save_list, f)









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='cfg file for training')
    parser.add_argument('--reload', action='store_true', help='whether to load existed model')
    parser.add_argument('--first_use_dataset', action='store_true', help='whether to load existed model')
    parser.add_argument('--causal', action='store_true', help='whether use causal version')
    parser.add_argument('--model_dir', type=str, default='None', help='if reload existed model, what is the model dir')
    parser.add_argument('--skip_to_use', type=str, default='012345', help='sc to use')
    parser.add_argument('--recon_loss_weight', type=str, default='111', help='what nsvae structure to use')
    parser.add_argument('--recon_type', type=str, default='real_imag', help='what recon method use')
    parser.add_argument('--resynthesis', action='store_true', help="whether use resynthesis calculate loss")
    parser.add_argument('--data_norm', action='store_true', help="whether do data normalization after STFT")

    args = parser.parse_args()
    cfg = myconf()
    cfg.read(args.cfg_file)

    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')

    data_mean = np.loadtxt(mean_file)
    data_mean = data_mean.reshape(1, data_mean.shape[0], 1, data_mean.shape[1])
    data_std = np.loadtxt(std_file)
    data_std = data_std.reshape(1, data_std.shape[0], 1, data_std.shape[1])

    log_params = {
        'reload': args.reload,
        'cfg_file': args.cfg_file,
        'model_dir': args.model_dir,
        'skip_to_use': args.skip_to_use,
        'recon_loss_weight': args.recon_loss_weight,
        'recon_type': args.recon_type,
        'first_use_dataset': args.first_use_dataset,
        'causal': args.causal,
        'resynthesis': args.resynthesis,
        'data_mean': data_mean,
        'data_std': data_std,
        'data_norm': args.data_norm

    }



    supervised_dccrn(cfg, log_params)
























