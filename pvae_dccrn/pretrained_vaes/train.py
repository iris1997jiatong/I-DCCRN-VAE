from __future__ import print_function

import numpy as np

import torch
import torch.optim as optim
import sys
import os

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(main_folder_path)
from model.pvae_module import *
from model.pretrain_pvaes_loss import complex_standard_vae_loss, KL_annealing
from utils.logger import get_logger

import argparse
import datetime

from utils.read_config import myconf
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig
from dataset import dataload_pretrained_vaes
import os
import shutil
import socket
import pickle

# torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Number of GPUs available:", torch.cuda.device_count())
# # Print the current device
print("Current device:", torch.cuda.current_device())

def check_and_log_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        # print(input)
        # print(f"Tensor contents:\n{tensor}")
        # self.detect_anormal = False
        raise RuntimeError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"inf detected in {name}")
        # print(input)
        # print(f"Tensor contents:\n{tensor}")
        # self.detect_anormal = False
        raise RuntimeError(f"inf detected in {name}")

    
def GRU_VAE_Train(cfg, log_params):

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
    
    
    # load model
    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')    
    num_samples = log_params['num_samples']
    zdim = log_params['zdim']
    skipc = log_params['skipc']
    skip_to_use = log_params['skip_to_use']
    skip_padding = log_params['skip_padding']
    recon_type = log_params['recon_type']
    recon_loss_type = log_params['recon_loss_type']
    recon_loss_weight = log_params['recon_loss_weight']
    causal = log_params['causal']
    fclatent = log_params['fclatent']
    if log_params['data_norm']:
        data_mean = log_params['data_mean']
        data_mean = torch.tensor(data_mean, dtype=torch.float32).to(device)
        data_std = log_params['data_std']
        data_std = torch.tensor(data_std, dtype=torch.float32).to(device)
    else:
        data_mean = None
        data_std = None 
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params() 

    if not skipc:
        if not fclatent:
            if not skip_padding:
                model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, data_mean, data_std)
                model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, data_mean=data_mean, data_std=data_std)
            else:
                model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skip_to_use)
        else:
            if not skip_padding:
                model_encoder = pvae_dccrn_encoder_no_skip_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, data_mean, data_std)
                model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, data_mean=data_mean, data_std=data_std)
            else:
                model_encoder = pvae_dccrn_encoder_skip_prepare_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skip_to_use)

    else:
        model_encoder = pvae_dccrn_encoder(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
        model_decoder = pvae_dccrn_decoder(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skip_to_use)
    basic_info.append('encoder params: %.2fM' % (sum(p.numel() for p in model_encoder.parameters()) / 1000000.0))
    basic_info.append('decoder params: %.2fM' % (sum(p.numel() for p in model_decoder.parameters()) / 1000000.0))

    model_encoder.to(device)
    model_decoder.to(device)

    learning_rate = cfg.getfloat('Training', 'lr')

    optimizer_en = optim.Adam(model_encoder.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_en, 'min', factor=0.5,patience=3)
    optimizer_de = optim.Adam(model_decoder.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_de, 'min', factor=0.5,patience=3)

    # define loss
    kl_anneal_flag = log_params['kl_ann_flag']
    kl_warm_epochs = log_params['kl_warm_epochs']
    kl_weight = log_params['kl_weight']
    prior_mode = log_params['prior_mode']
    if kl_anneal_flag:
        kl_warm = KL_annealing(kl_warm_epochs)
        kl_warm_weights = kl_warm.frange_cycle_linear(stop=kl_weight)
    else:
        kl_warm_weights = kl_weight * torch.ones(kl_warm_epochs)
    kl_warm_weights = kl_warm_weights.to(device)
    mi_weight = log_params['mi_weight']
    loss_train = complex_standard_vae_loss(kl_warm_weights, kl_weight, mi_weight, recon_loss_type, recon_type, recon_loss_weight, num_samples, prior_mode)

    if not log_params['reload']:
        saved_root = cfg.get('User', 'saved_root')
        filename = "{}_{}_causal={}_zdim={}_numsamples={}_klw={:.3f}_miw={}_skipc={}_skipuse={}_spadd={}_recon={}_reconweight={}_prior={}".format(date, model_name, causal, zdim, num_samples, kl_weight, mi_weight,
                                                                                                       skipc, skip_to_use, skip_padding, recon_type, recon_loss_weight, prior_mode)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)
    else:
    #     tag = self.cfg.get('Network', 'tag')
        filename = "{}_{}_causal={}_zdim={}_numsamples={}_klw={:.3f}_miw={}_skipc={}_skipuse={}_spadd={}_recon={}_reconweight={}_prior={}".format(date, model_name, causal, zdim, num_samples, kl_weight, mi_weight,
                                                                                                       skipc, skip_to_use, skip_padding, recon_type, recon_loss_weight, prior_mode)
        save_dir = log_params['reload_savedir']

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
    train_dataloader, val_dataloader, train_num, val_num = dataload_pretrained_vaes.build_dataloader(cfg, first_use_dataset)
    logger.info('Train on {}'.format(dataset_name))
    logger.info('Training samples: {}'.format(train_num))
    logger.info('Validation samples: {}'.format(val_num))

    # load training parameters
    epochs = cfg.getint('Training', 'epochs')
    early_stop_patience = cfg.getint('Training', 'early_stop_patience')
    save_frequency = cfg.getint('Training', 'save_frequency')


    # Create python list for loss
    if not log_params['reload']:
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_kl = np.zeros((epochs,))
        train_mi = np.zeros((epochs,))
        train_cpx = np.zeros((epochs,))
        train_mag = np.zeros((epochs,))
        train_sisnr = np.zeros((epochs,))

        val_recon = np.zeros((epochs,))
        val_kl = np.zeros((epochs,))
        val_mi = np.zeros((epochs,))
        val_cpx = np.zeros((epochs,))
        val_mag = np.zeros((epochs,))
        val_sisnr = np.zeros((epochs,))

        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_encoder_state_dict = model_encoder.state_dict()
        best_decoder_state_dict = model_decoder.state_dict()
        best_encoder_optim_dict = optimizer_en.state_dict()
        best_decoder_optim_dict = optimizer_de.state_dict()
        best_encoder_scheduler_dict = scheduler_encoder.state_dict()
        best_decoder_scheduler_dict = scheduler_decoder.state_dict()
        start_epoch = -1
    else:
        cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(model_name))
        checkpoint = torch.load(cp_file)
        model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_en.load_state_dict(checkpoint['encoder_optim_state_dict'])
        optimizer_de.load_state_dict(checkpoint['decoder_optim_state_dict'])
        scheduler_encoder.load_state_dict(checkpoint['encoder_scheduler'])
        scheduler_decoder.load_state_dict(checkpoint['decoder_scheduler'])
        start_epoch = checkpoint['epoch']
        loss_log = checkpoint['loss_log']
        train_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_recon = np.pad(loss_log['train_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_kl = np.pad(loss_log['train_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_mi = np.pad(loss_log['train_mi'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_cpx = np.pad(loss_log['train_cpx'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_mag = np.pad(loss_log['train_mag'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        train_sisnr = np.pad(loss_log['train_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_kl = np.pad(loss_log['val_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_mi = np.pad(loss_log['val_mi'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_cpx = np.pad(loss_log['val_cpx'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_mag = np.pad(loss_log['val_mag'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        val_sisnr = np.pad(loss_log['val_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)

        best_val_loss = checkpoint['best_val_loss']
        cpt_patience = checkpoint['cpt_patience']
        cur_best_epoch = start_epoch
        best_encoder_state_dict = model_encoder.state_dict()
        best_decoder_state_dict = model_decoder.state_dict()
        best_encoder_optim_dict = optimizer_en.state_dict()
        best_decoder_optim_dict = optimizer_de.state_dict()
        best_encoder_scheduler_dict = scheduler_encoder.state_dict()
        best_decoder_scheduler_dict = scheduler_decoder.state_dict()
        logger.info('Resuming trainning: epoch: {}'.format(start_epoch))


    for epoch in range(start_epoch+1, epochs):
        running_loss_epoch_train = 0.0
        loss_kl_epoch_train = 0.0
        loss_mi_epoch_train = 0.0
        loss_recon_epoch_train = 0.0
        loss_cpx_epoch_train = 0.0
        loss_mag_epoch_train = 0.0
        loss_sisnr_epoch_train = 0.0

        running_loss_epoch_val = 0.0
        loss_kl_epoch_val = 0.0
        loss_mi_epoch_val = 0.0
        loss_recon_epoch_val = 0.0
        loss_cpx_epoch_val = 0.0
        loss_mag_epoch_val = 0.0
        loss_sisnr_epoch_val = 0.0

        # training
        model_encoder.train()
        model_decoder.train()
        start_time = datetime.datetime.now()
        for i, batch_data in enumerate(train_dataloader):
                
            batch_data = batch_data.to(device) # B, time len
            bs, time_len = batch_data.shape

            batch_data = batch_data.float()
            z, miu, log_sigma, delta, skiper, C, F, stft_x = model_encoder(batch_data, train=True)
            recon_sig, predict_stft = model_decoder(stft_x, z, skiper, C, F, train=True) # B * num_samples, time len
            batch_data = batch_data.unsqueeze(1) # B, 1, time len
            batch_data = batch_data.repeat(1,num_samples,1)
            batch_data = batch_data.view(bs * num_samples, time_len)

            stft_x = stft_x.unsqueeze(1) # B, 1, F; T
            freq, time = stft_x.shape[2], stft_x.shape[3]
            stft_x = stft_x.repeat(1,num_samples,1, 1, 1)
            stft_x = stft_x.view(bs * num_samples, freq, time, 2)            

            loss, loss_recon, loss_kl, loss_mi, loss_cpx, loss_mag, sisnr = loss_train.cal_loss(batch_data, recon_sig,stft_x, predict_stft, miu, log_sigma, delta, z, epoch)


            optimizer_en.zero_grad()
            optimizer_de.zero_grad()
            loss.backward()
            

            optimizer_de.step()
            optimizer_en.step()

            # loss per time frame
            running_loss_epoch_train += loss.item() * bs
            loss_kl_epoch_train += loss_kl.item() * bs
            loss_mi_epoch_train += loss_mi.item() * bs
            loss_recon_epoch_train += loss_recon.item() * bs
            loss_cpx_epoch_train += loss_cpx.item() * bs
            loss_mag_epoch_train += loss_mag.item() * bs
            loss_sisnr_epoch_train += sisnr.item() * bs


        # loss per sample
        train_loss[epoch] = running_loss_epoch_train / train_num
        train_recon[epoch] = loss_recon_epoch_train / train_num
        train_kl[epoch] = loss_kl_epoch_train / train_num
        train_mi[epoch] = loss_mi_epoch_train / train_num

        train_cpx[epoch] = loss_cpx_epoch_train / train_num
        train_mag[epoch] = loss_mag_epoch_train / train_num
        train_sisnr[epoch] = loss_sisnr_epoch_train / train_num



        # validation
        model_encoder.eval()
        model_decoder.eval()
        for i, batch_data in enumerate(val_dataloader):

            batch_data = batch_data.to(device) # B, time len
            bs, time_len = batch_data.shape

            batch_data = batch_data.float()
            with torch.no_grad():
                z, miu, log_sigma, delta, skiper, C, F, stft_x = model_encoder(batch_data, train=False)
                recon_sig, predict_stft = model_decoder(stft_x, z, skiper, C, F, train=False) # B * num_samples, time len

            batch_data = batch_data.unsqueeze(1) # B, 1, time len
            batch_data = batch_data.repeat(1,num_samples,1)
            batch_data = batch_data.view(bs * num_samples, time_len)

            stft_x = stft_x.unsqueeze(1) # B, 1, F; T, 2
            freq, time = stft_x.shape[2], stft_x.shape[3]
            stft_x = stft_x.repeat(1,num_samples,1, 1, 1)
            stft_x = stft_x.view(bs * num_samples, freq, time, 2)            

            loss, loss_recon, loss_kl, loss_mi, loss_cpx, loss_mag, sisnr = loss_train.cal_loss(batch_data, recon_sig,stft_x, predict_stft, miu, log_sigma, delta, z, kl_warm_epochs+2)

            # loss per time frame
            running_loss_epoch_val += loss.item() * bs
            loss_kl_epoch_val += loss_kl.item() * bs
            loss_mi_epoch_val += loss_mi.item() * bs
            loss_recon_epoch_val += loss_recon.item() * bs
            loss_cpx_epoch_val += loss_cpx.item() * bs
            loss_mag_epoch_val += loss_mag.item() * bs
            loss_sisnr_epoch_val += sisnr.item() * bs
        # loss per sample
        val_loss[epoch] = running_loss_epoch_val / val_num
        val_recon[epoch] = loss_recon_epoch_val / val_num
        val_kl[epoch] = loss_kl_epoch_val / val_num
        val_mi[epoch] = loss_mi_epoch_val / val_num
        val_cpx[epoch] = loss_cpx_epoch_val / val_num
        val_mag[epoch] = loss_mag_epoch_val / val_num
        val_sisnr[epoch] = loss_sisnr_epoch_val / val_num

        scheduler_encoder.step(val_loss[epoch])
        scheduler_decoder.step(val_loss[epoch]) 

        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            cpt_patience = 0
            best_encoder_state_dict = model_encoder.state_dict()
            best_decoder_state_dict = model_decoder.state_dict()
            best_encoder_optim_dict = optimizer_en.state_dict()
            best_decoder_optim_dict = optimizer_de.state_dict()
            best_encoder_scheduler_dict = scheduler_encoder.state_dict()
            best_decoder_scheduler_dict = scheduler_decoder.state_dict()
            cur_best_epoch = epoch

            save_file = os.path.join(save_dir, model_name + '_encoder_best_epoch.pt')
            torch.save(best_encoder_state_dict, save_file)

            save_file = os.path.join(save_dir, model_name + '_decoder_best_epoch.pt')
            torch.save(best_decoder_state_dict, save_file)

            loss_log = {'train_loss': train_loss[:cur_best_epoch+1],
                        'val_loss': val_loss[:cur_best_epoch+1],
                        'train_recon': train_recon[:cur_best_epoch+1],
                        'train_kl': train_kl[:cur_best_epoch+1],
                        'train_mi': train_mi[:cur_best_epoch+1],
                        'train_cpx':train_cpx[:cur_best_epoch+1],
                        'train_mag':train_mag[:cur_best_epoch+1],
                        'train_sisnr': train_sisnr[:cur_best_epoch+1],
                        'val_recon': val_recon[:cur_best_epoch+1],
                        'val_kl': val_kl[:cur_best_epoch+1],
                        'val_mi': val_mi[:cur_best_epoch+1],
                        'val_cpx':val_cpx[:cur_best_epoch+1],
                        'val_mag':val_mag[:cur_best_epoch+1],
                        'val_sisnr': val_sisnr[:cur_best_epoch+1],
                }

            save_file = os.path.join(save_dir, model_name + '_checkpoint.pt')
            save_dict = {'epoch': cur_best_epoch,
                        'best_val_loss': best_val_loss,
                        'cpt_patience': cpt_patience,
                        'encoder_state_dict': best_encoder_state_dict,
                        'decoder_state_dict': best_decoder_state_dict,
                        'loss_log': loss_log
                }
            save_dict['encoder_optim_state_dict'] = best_encoder_optim_dict
            save_dict['decoder_optim_state_dict'] = best_decoder_optim_dict
            save_dict['encoder_scheduler'] = best_encoder_scheduler_dict
            save_dict['decoder_scheduler'] = best_decoder_scheduler_dict
            torch.save(save_dict, save_file)
            
            logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))
        else:
            cpt_patience += 1

        # Training time
        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
        logger.info('Train => tot: {:.2f} recon {:.2f} cpx {:.2f} mag {:.2f} sisnr {:.2f} KL {:.2f} mi {:.2f}'.format(
                                                                                                train_loss[epoch], train_recon[epoch],
                                                                                                train_cpx[epoch], train_mag[epoch], train_sisnr[epoch], train_kl[epoch], train_mi[epoch]))
        logger.info('val => tot: {:.2f} recon {:.2f} cpx {:.2f} mag {:.2f} sisnr {:.2f} KL {:.2f} mi {:.2f}'.format(
                                                                                                val_loss[epoch], val_recon[epoch],
                                                                                                val_cpx[epoch], val_mag[epoch], val_sisnr[epoch], val_kl[epoch], val_mi[epoch]))   

        # Stop traning if early-stop triggers
        if cpt_patience == early_stop_patience:
            logger.info('Early stop patience achieved')
            break   
    
    
    # stop neptune logger
    # run.stop()


    # Save the training loss and validation loss
    train_loss = train_loss[:epoch+1]
    val_loss = val_loss[:epoch+1]
    train_recon = train_recon[:epoch+1]
    train_kl = train_kl[:epoch+1]
    train_mi = train_mi[:epoch+1]
    train_cpx = train_cpx[:epoch+1]
    train_mag = train_mag[:epoch+1]
    train_sisnr = train_sisnr[:epoch+1]
    val_recon = val_recon[:epoch+1]
    val_kl = val_kl[:epoch+1]
    val_mi = val_mi[:epoch+1]
    val_cpx = val_cpx[:epoch+1]
    val_mag = val_mag[:epoch+1]
    val_sisnr = val_sisnr[:epoch+1]
    loss_file = os.path.join(save_dir, 'loss_model.pckl')
    with open(loss_file, 'wb') as f:
        pickle.dump([train_loss, val_loss, train_recon, train_kl, train_mi, train_cpx, train_mag, train_sisnr,
                      val_recon, val_kl, val_mi, val_cpx, val_mag, val_sisnr], f)
    loss_title_file = os.path.join(save_dir, 'loss_titles.pckl')
    with open(loss_title_file, 'wb') as f:
        pickle.dump(['train loss', 'val loss', 'recon train', 'kl train', 'kl mi', 'cpx train','mag train','sisnr train',
                      'recon val', 'kl val','mi val','cpx val','mag val','sisnr val'] ,f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='cfg file for training')
    parser.add_argument('--first_use_dataset', action="store_true", help='whether a new dataset to train')
    parser.add_argument('--causal', action='store_true', help='whether use causal version')
    parser.add_argument('--reload', action="store_true", help='whether to load existed model')
    parser.add_argument('--reload_savedir', type=str, default=None, help='which model to reload')
    parser.add_argument('--zdim', type=int, default=128, help='dimension of the latent space')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--kl_ann_flag', action="store_true", help='whether to use kl annealing')
    parser.add_argument('--kl_warm_epochs', type=int, default=20,  help='num of epochs of kl warm')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='weight of kl div term in loss function')
    parser.add_argument('--mi_weight', type=float, default=0.0, help='weight of MI term in loss function')
    parser.add_argument('--skipc',  action="store_true", help='whether use skip connection')
    parser.add_argument('--fclatent', action='store_true', help='whether use fc latent')
    parser.add_argument('--skip_to_use', type=str, default='012345', help='where to use skip connection')
    parser.add_argument('--skip_padding', action='store_true', help='whether padding zeros as skipc')
    parser.add_argument('--recon_type', type=str, default='real_imag', help='what is reconstructed in decoder output')
    parser.add_argument('--recon_loss_type', type=str, default='multiple', help='what loss is used for recon loss')
    parser.add_argument('--recon_loss_weight', type=str, default='1.0,1.0,0.0', help='weight for cpx part, mag part and sisnr')
    parser.add_argument('--prior_mode', type=str, default='ri_inde', help='which p(z) to assume')
    parser.add_argument('--data_norm', action='store_true', help="whether do data normalization after STFT")

    args = parser.parse_args()
    cfg = myconf()
    cfg.read(args.cfg_file)

    skip_to_use = []
    for i in args.skip_to_use:
        skip_to_use.append(int(i))
    print(skip_to_use)   
    recon_loss_weight = []
    weights = args.recon_loss_weight.split(',')
    for w in weights:
        w = float(w)
        recon_loss_weight.append(w)
    print(recon_loss_weight)

    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')

    data_mean = np.loadtxt(mean_file)
    data_mean = data_mean.reshape(1, data_mean.shape[0], 1, data_mean.shape[1])
    data_std = np.loadtxt(std_file)
    data_std = data_std.reshape(1, data_std.shape[0], 1, data_std.shape[1])    
    log_params = {
         'reload': args.reload,
         'reload_savedir': args.reload_savedir,
         'cfg_file': args.cfg_file,
         'first_use_dataset': args.first_use_dataset,
         'causal': args.causal,
         'zdim': args.zdim,
         'num_samples':args.num_samples,
         'kl_ann_flag': args.kl_ann_flag,
         'kl_warm_epochs': args.kl_warm_epochs,
         'kl_weight': args.kl_weight,
         'mi_weight': args.mi_weight,
         'skipc': args.skipc,
         'fclatent': args.fclatent,
         'skip_to_use': skip_to_use,
         'skip_padding': args.skip_padding,
         'recon_type': args.recon_type,
         'recon_loss_type': args.recon_loss_type,
         'recon_loss_weight': recon_loss_weight,
         'prior_mode': args.prior_mode,
         'data_mean': data_mean,
         'data_std': data_std,
         'data_norm': args.data_norm
    }

    GRU_VAE_Train(cfg, log_params)

















