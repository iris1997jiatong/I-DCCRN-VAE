from __future__ import print_function

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

torch.manual_seed(5785)
torch.cuda.manual_seed(42132)
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
#torch.manual_seed(333)
# torch.cuda.manual_seed(333)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(main_folder_path)

from model.pvae_module import *
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig
from model.nsvae_loss import *
from dataset import dataload_nsvae
import datetime
import argparse
import socket
import os
import shutil
from utils.logger import get_logger
import pickle
from utils.read_config import myconf


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def check_and_log_nan(tensor, name):
    if tensor is not None:
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"inf detected in {name}")
            raise RuntimeError(f"inf detected in {name}")




def train_nsvae_decoder(cfg, log_params, best_clean_encoder, best_clean_decoder, best_noise_encoder, best_noise_decoder, best_noisy_encoder, skipuse):
    
    # get basic info
    date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
    hostname = socket.gethostname()
    model_name = cfg.get('User', 'model_name') + '_adv_net'
    dataset_name = cfg.get('DataFrame', 'dataset_name')


    basic_info = []
    basic_info.append('HOSTNAME: ' + hostname)
    basic_info.append('Time: ' + date)
    basic_info.append('model name (phase 2):' + model_name)
    basic_info.append('Device for training: ' + device)
    if device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))

    # load training parameters
    epochs = cfg.getint('Training', 'epochs')
    early_stop_patience = cfg.getint('Training', 'early_stop_patience')

    learning_rate = cfg.getfloat('Training', 'lr')

    # load model
    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')    
    num_samples = log_params['num_samples']
    zdim = log_params['zdim']
    causal = log_params['causal']
    latent_num = log_params['latent_num']
    pre_latent_num = log_params['pre_latent_num']
    skipuse_str = log_params['skipuse_str']
    use_sc_phase2 = log_params['use_sc_phase2']
    load_de = log_params['load_de']
    recon_type = log_params['recon_type']
    resynthesis = log_params['resynthesis']
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()

    # nsvae model params
    noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase(net_params, causal, device, zdim, nfft, hop, wlen, num_samples, pre_latent_num)
    noisy_model_encoder.load_state_dict(best_noisy_encoder)
    noisy_model_encoder.to(device)
    # noisy_model_encoder = torch.compile(noisy_model_encoder)
    for param in noisy_model_encoder.parameters():
        param.requires_grad = False
    basic_info.append('noisy encoder params: %.2fM' % (sum(p.numel() for p in noisy_model_encoder.parameters()) / 1000000.0))
    # if latent_num == 1:
    noisy_clean_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, use_sc=use_sc_phase2, skip_to_use=skipuse, resynthesis=resynthesis) 
    if load_de:
        noisy_clean_model_decoder.load_state_dict(best_clean_decoder)
    noisy_clean_model_decoder.to(device)
    speech_distin = distinguisher(net_params, causal, device, zdim, nfft, hop, wlen)
    speech_distin.to(device)
    basic_info.append('noisy decoder params: %.2fM' % (sum(p.numel() for p in noisy_clean_model_decoder.parameters()) / 1000000.0))  
    model_loss = adversarial_second_phase_loss(latent_num)
    

    if log_params['decode_update'] == 'all_decode':
        optimizer_noisy_clean_de = optim.Adam(noisy_clean_model_decoder.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler_noisy_clean_de = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noisy_clean_de, 'min', factor=0.5,patience=3)
    elif log_params['decode_update'] == 'skip_layer':
        for param in noisy_clean_model_decoder.parameters():
            param.requires_grad = False
        for skip in skipuse:
            decode_id = int(len(noisy_clean_model_decoder.decoders) - skip - 1)
            for param in noisy_clean_model_decoder.decoders[decode_id].parameters():
                param.requires_grad = True  # Unfreeze last decoder
        optimizer_noisy_clean_de = optim.Adam(filter(lambda p: p.requires_grad, noisy_clean_model_decoder.parameters()), lr=learning_rate, weight_decay=0.001)
        scheduler_noisy_clean_de = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noisy_clean_de, 'min', factor=0.5,patience=3)                
    optimizer_speech_dis = optim.Adam(speech_distin.parameters(), lr=log_params['dlr'], weight_decay=0.001)
    scheduler_speech_dis = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_speech_dis, 'min', factor=0.5,patience=3)

    if not log_params['reload']:
            decode_update = log_params['decode_update']
            saved_root = cfg.get('User', 'saved_root')
            dlr_str = f"{log_params['dlr']:.0e}".replace('e-0', 'e-')
            filename = "{}_{}_causal={}_zdim={}_latentnum={}_decodeupdate={}_skipc={}_skipuse={}_numsamples={}_loadde={}_recontype={}_resyn={}_dlr={}_dstep={}".format(date, model_name, causal, zdim, latent_num, decode_update, use_sc_phase2, skipuse_str, num_samples, load_de, recon_type, resynthesis, 
                                                                                                                                                                       dlr_str, log_params['d_step'])
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
    save_log_phase1 = os.path.join(log_params['first_phase_folder'], 'log.txt')
    target = os.path.join(save_dir, 'log_phase1.txt')
    shutil.copy(save_log_phase1, target)
    save_file = os.path.join(save_dir, model_name + '_clean_decoder_best_epoch.pt')
    torch.save(best_clean_decoder, save_file)
    save_file = os.path.join(save_dir, model_name + '_noise_decoder_best_epoch.pt')
    torch.save(best_noise_decoder, save_file)

    save_file = os.path.join(save_dir, model_name + '_clean_encoder_best_epoch.pt')
    torch.save(best_clean_encoder, save_file)
    save_file = os.path.join(save_dir, model_name + '_noise_encoder_best_epoch.pt')
    torch.save(best_noise_encoder, save_file)
    # create logger
    log_file = os.path.join(save_dir, 'log_phase2.txt')
    logger_type = cfg.getint('User', 'logger_type')
    logger = get_logger(log_file, logger_type)

    # Print basical infomation
    for log in basic_info:
            logger.info(log)
    logger.info('In this experiment, result will be saved in: ' + save_dir)     
    logger.info('first phase folder---' + log_params['first_phase_folder'])



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
        epoch_train_noisy_recon_loss_sisnr = np.zeros((epochs,))
        epoch_train_dis_loss = np.zeros((epochs,))
        epoch_train_gen_dis_loss = np.zeros((epochs,))

        # val loss
        epoch_val_total_loss = np.zeros((epochs,))
        epoch_val_noisy_recon_loss_sisnr = np.zeros((epochs,))
        epoch_val_dis_loss = np.zeros((epochs,))
        epoch_val_gen_dis_loss = np.zeros((epochs,))

        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()
        # if latent_num == 1:
        best_noisy_clean_decoder_state_dict = noisy_clean_model_decoder.state_dict()
        best_noisy_clean_decoder_optim_dict = optimizer_noisy_clean_de.state_dict()
        best_noisy_clean_decoder_scheduler_dict = scheduler_noisy_clean_de.state_dict()
        best_speech_dis_state_dict = speech_distin.state_dict()
        best_speech_dis_optim_dict = optimizer_speech_dis.state_dict()
        best_speech_dis_scheduler_dict = scheduler_speech_dis.state_dict()


        start_epoch = -1
    else:
        # resume training from certain epoch
        # load the model
        cp_file = os.path.join(save_dir, '{}_checkpoint_phase2.pt'.format(model_name))
        checkpoint = torch.load(cp_file)
        noisy_model_encoder.load_state_dict(checkpoint['noisy_encoder_state_dict'])
        # if latent_num == 1:
        noisy_clean_model_decoder.load_state_dict(checkpoint['noisy_clean_decoder_state_dict'])
        optimizer_noisy_clean_de.load_state_dict(checkpoint['noisy_clean_decoder_optim_dict'])
        scheduler_noisy_clean_de.load_state_dict(checkpoint['noisy_clean_decoder_scheduler_dict'])
        speech_distin.load_state_dict(checkpoint['speech_distin_state_dict'])
        optimizer_speech_dis.load_state_dict(checkpoint['speech_dis_optim_dict'])
        scheduler_speech_dis.load_state_dict(checkpoint['speech_dis_scheduler_dict'])

        start_epoch = checkpoint['epoch']
        loss_log = checkpoint['loss_log']

        # load the loss
        epoch_train_total_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_total_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_noisy_recon_loss_sisnr = np.pad(loss_log['train_noisy_recon_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_dis_loss = np.pad(loss_log['train_dis_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_train_gen_dis_loss = np.pad(loss_log['train_gen_dis_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_noisy_recon_loss_sisnr = np.pad(loss_log['val_noisy_recon_sisnr'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_dis_loss = np.pad(loss_log['val_dis_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
        epoch_val_gen_dis_loss = np.pad(loss_log['val_gen_dis_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)

        # load the current best model
        best_val_loss = checkpoint['best_val_loss']
        cpt_patience = checkpoint['cpt_patience']
        cur_best_epoch = start_epoch

        best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()
        # if latent_num == 1:
        best_noisy_clean_decoder_state_dict = noisy_clean_model_decoder.state_dict()
        best_noisy_clean_decoder_optim_dict = optimizer_noisy_clean_de.state_dict()
        best_noisy_clean_decoder_scheduler_dict = scheduler_noisy_clean_de.state_dict()
        best_speech_dis_state_dict = speech_distin.state_dict()
        best_speech_dis_optim_dict = optimizer_speech_dis.state_dict()
        best_speech_dis_scheduler_dict = scheduler_speech_dis.state_dict()


        logger.info('Resuming trainning: epoch: {}'.format(start_epoch))            




    for epoch in range(start_epoch+1, epochs):
        # loss
        train_total_loss = 0
        train_noisy_recon_loss_sisnr = 0
        train_dis_loss = 0
        train_gen_dis_loss = 0


        val_total_loss = 0
        val_noisy_recon_loss_sisnr = 0
        val_dis_loss = 0
        val_gen_dis_loss = 0

        # training


        noisy_model_encoder.eval()
        noisy_clean_model_decoder.train()
        speech_distin.train()


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


            z_noisy_speech, miu_noisy_speech, log_sigma_noisy_speech, delta_noisy_speech, z_noisy_noise, miu_noisy_noise, log_sigma_noisy_noise, delta_noisy_noise, skiper_noisy, C, F, stft_x = noisy_model_encoder(noisy_batch, train=False)
            
            recon_sig_clean, predict_stft_clean = noisy_clean_model_decoder(stft_x, z_noisy_speech, skiper_noisy, C, F, train=True, pad='sig') # B * num_samples, time len


            clean_batch = clean_batch.unsqueeze(1) # B, 1, time len
            clean_batch = clean_batch.repeat(1,num_samples,1)
            clean_batch = clean_batch.view(bs * num_samples, time_len)
            # predict_stft_noise = None
            # stft_x_noise = None
            noise_batch = None
            recon_sig_noise = None
            if i % log_params['d_step'] == 0:
                dist_true_clean = speech_distin(clean_batch, train=True)
                dist_est_clean = speech_distin(recon_sig_clean.detach(), train=True)
                dist_true_noise = None
                dist_est_noise = None
                # update D
                dist_loss = model_loss.distinguisher_loss(dist_true_clean, dist_est_clean, dist_true_noise, dist_est_noise)
                optimizer_speech_dis.zero_grad()
                dist_loss.backward() # <-- Keep the graph
                optimizer_speech_dis.step()
            dist_gen_est_clean = speech_distin(recon_sig_clean, train=True)     
            dist_gen_est_noise = None           

            # update G
            gen_loss, loss_recon, loss_dis_gen = model_loss.generator_loss(clean_batch, recon_sig_clean, dist_gen_est_clean, noise_batch, recon_sig_noise, dist_gen_est_noise)           

            optimizer_noisy_clean_de.zero_grad()
            gen_loss.backward()
            optimizer_noisy_clean_de.step()
            # loss per time frame
            train_total_loss += gen_loss.item() * bs
            train_noisy_recon_loss_sisnr += loss_recon.item() * bs
            train_dis_loss += dist_loss.item() * bs
            train_gen_dis_loss += loss_dis_gen.item() * bs

        # loss per sample per time frame
        epoch_train_total_loss[epoch] = train_total_loss / train_num     
        epoch_train_noisy_recon_loss_sisnr[epoch] = train_noisy_recon_loss_sisnr / train_num   
        epoch_train_dis_loss[epoch] = train_dis_loss / train_num
        epoch_train_gen_dis_loss[epoch] = train_gen_dis_loss / train_num      


        # validation
        noisy_model_encoder.eval()
        noisy_clean_model_decoder.eval()
        speech_distin.eval()
        for i, batch_data in enumerate(val_dataloader):

            noisy_batch, clean_batch, noise_batch = batch_data[0], batch_data[1], batch_data[2]
            noisy_batch = noisy_batch.to(device) # [batch, time, freq]
            clean_batch = clean_batch.to(device)
            noise_batch = noise_batch.to(device)

            bs, time_len = noisy_batch.shape

            noisy_batch = noisy_batch.float()
            clean_batch = clean_batch.float()
            noise_batch = noise_batch.float()

            with torch.no_grad():
                z_noisy_speech, miu_noisy_speech, log_sigma_noisy_speech, delta_noisy_speech, z_noisy_noise, miu_noisy_noise, log_sigma_noisy_noise, delta_noisy_noise, skiper_noisy, C, F, stft_x = noisy_model_encoder(noisy_batch, train=False)
                
                recon_sig_clean, predict_stft_clean = noisy_clean_model_decoder(stft_x, z_noisy_speech, skiper_noisy, C, F, train=False, pad='sig') # B * num_samples, time len

                clean_batch = clean_batch.unsqueeze(1) # B, 1, time len
                clean_batch = clean_batch.repeat(1,num_samples,1)
                clean_batch = clean_batch.view(bs * num_samples, time_len)
                
                noise_batch = None
                recon_sig_noise = None
                dist_true_clean = speech_distin(clean_batch, train=True)
                dist_est_clean = speech_distin(recon_sig_clean, train=True)
                dist_true_noise = None
                dist_est_noise = None

                # update D
                dist_loss = model_loss.distinguisher_loss(dist_true_clean, dist_est_clean, dist_true_noise, dist_est_noise)
                # update G
                gen_loss, loss_recon, loss_dis_gen = model_loss.generator_loss(clean_batch, recon_sig_clean, dist_est_clean, noise_batch, recon_sig_noise, dist_est_noise)      

            # loss
            # loss per time frame
            val_total_loss += gen_loss.item() * bs
            val_noisy_recon_loss_sisnr += loss_recon.item() * bs
            val_dis_loss += dist_loss.item() * bs
            val_gen_dis_loss += loss_dis_gen.item() * bs

        # loss per sample per time frame
        epoch_val_total_loss[epoch] = val_total_loss / val_num     
        epoch_val_noisy_recon_loss_sisnr[epoch] = val_noisy_recon_loss_sisnr / val_num   
        epoch_val_dis_loss[epoch] = val_dis_loss / val_num
        epoch_val_gen_dis_loss[epoch] = val_gen_dis_loss / val_num   
 
        scheduler_noisy_clean_de.step(epoch_val_noisy_recon_loss_sisnr[epoch])
        scheduler_speech_dis.step(epoch_val_dis_loss[epoch])

            



        # save the current best model according to total val loss
        if epoch_val_noisy_recon_loss_sisnr[epoch] < best_val_loss:
            best_val_loss = epoch_val_noisy_recon_loss_sisnr[epoch]
            cpt_patience = 0
            best_noisy_encoder_state_dict = noisy_model_encoder.state_dict()
            save_file = os.path.join(save_dir, model_name + '_noisy_encoder_best_epoch.pt')
            torch.save(best_noisy_encoder_state_dict, save_file)

            best_noisy_clean_decoder_state_dict = noisy_clean_model_decoder.state_dict()
            best_noisy_clean_decoder_optim_dict = optimizer_noisy_clean_de.state_dict()
            best_noisy_clean_decoder_scheduler_dict = scheduler_noisy_clean_de.state_dict()
            best_speech_dis_state_dict = speech_distin.state_dict()
            best_speech_dis_optim_dict = optimizer_speech_dis.state_dict()
            best_speech_dis_scheduler_dict = scheduler_speech_dis.state_dict()
            save_file = os.path.join(save_dir, model_name + '_noisy_clean_decoder_best_epoch.pt')
            torch.save(best_noisy_clean_decoder_state_dict, save_file)



            cur_best_epoch = epoch

            loss_log = {'train_loss': epoch_train_total_loss[:cur_best_epoch+1],
                        'val_loss': epoch_val_total_loss[:cur_best_epoch+1],
                        'train_noisy_recon_sisnr': epoch_train_noisy_recon_loss_sisnr[:cur_best_epoch+1],
                        'train_dis_loss': epoch_train_dis_loss[:cur_best_epoch+1],
                        'train_gen_dis_loss': epoch_train_gen_dis_loss[:cur_best_epoch+1],
                        'val_noisy_recon_sisnr': epoch_val_noisy_recon_loss_sisnr[:cur_best_epoch+1],
                        'val_dis_loss': epoch_val_dis_loss[:cur_best_epoch+1],
                        'val_gen_dis_loss': epoch_val_gen_dis_loss[:cur_best_epoch+1]

                }

            save_file = os.path.join(save_dir, model_name + '_checkpoint_phase2.pt')
            # if latent_num == 1:
            save_dict = {'epoch': cur_best_epoch,
                        'best_val_loss': best_val_loss,
                        'cpt_patience': cpt_patience,
                        'noisy_encoder_state_dict': best_noisy_encoder_state_dict,
                        'noisy_clean_decoder_state_dict': best_noisy_clean_decoder_state_dict,
                        'speech_dist': best_speech_dis_state_dict,
                        'loss_log': loss_log
                }
            
            save_dict['noisy_clean_decoder_optim_dict'] = best_noisy_clean_decoder_optim_dict
            save_dict['noisy_clean_decoder_scheduler_dict'] = best_noisy_clean_decoder_scheduler_dict
            save_dict['speech_dist_optim_dict'] = best_speech_dis_optim_dict
            save_dict['speech_dist_scheduler_dict'] = best_speech_dis_scheduler_dict
            torch.save(save_dict, save_file)
    
            logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))    
        else:
            cpt_patience += 1   

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
        logger.info('Train => gen: {:.2f} recon sisnr {:.2f} gendis {:.4f} disloss {:.4f}'.format(epoch_train_total_loss[epoch], epoch_train_noisy_recon_loss_sisnr[epoch],
                                                                                                  epoch_train_gen_dis_loss[epoch], epoch_train_dis_loss[epoch]))
        logger.info('val => gen: {:.2f} recon sisnr {:.2f} gendis {:.4f} disloss {:.4f}'.format(epoch_val_total_loss[epoch], epoch_val_noisy_recon_loss_sisnr[epoch],
                                                                                                epoch_val_gen_dis_loss[epoch], epoch_val_dis_loss[epoch]))
        # Stop traning if early-stop triggers
        if cpt_patience == early_stop_patience:
            logger.info('Early stop patience achieved')
            break 


    # Save the training loss and validation loss
    train_loss_log = epoch_train_total_loss[:epoch+1]
    train_noisy_recon_log_sisnr = epoch_train_noisy_recon_loss_sisnr[:epoch+1]
    train_gen_dis_log = epoch_train_gen_dis_loss[:epoch+1]
    train_dis_log = epoch_val_dis_loss[:epoch+1]


    val_loss_log = epoch_val_total_loss[:epoch+1]
    val_noisy_recon_log_sisnr = epoch_val_noisy_recon_loss_sisnr[:epoch+1]
    val_gen_dis_log = epoch_val_gen_dis_loss[:epoch+1]
    val_dis_log = epoch_val_dis_loss[:epoch+1]


    loss_file = os.path.join(save_dir, 'loss_model_phase2.pckl')
    with open(loss_file, 'wb') as f:
        save_list = [train_loss_log, train_noisy_recon_log_sisnr, train_gen_dis_log, train_dis_log,
                    val_loss_log, val_noisy_recon_log_sisnr, val_gen_dis_log, val_dis_log]
        pickle.dump(save_list, f)











if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='cfg file for training')
    parser.add_argument('--reload', action='store_true', help='whether to load existed model')
    parser.add_argument('--model_dir', type=str, default='None', help='if reload existed model, what is the model dir')
    parser.add_argument('--first_use_dataset', action='store_true', help='first use this dataset?')
    parser.add_argument('--first_phase_folder', type=str, help='the folder to the models trained in the first phase')
    parser.add_argument('--causal', action='store_true', help='use causal version?')   
    parser.add_argument('--decode_update', type=str, default='all_decode', help="which part of decoder get updated")
    parser.add_argument('--use_sc_phase2', action='store_true', help='whether to use sc in phase 2')
    parser.add_argument('--num_samples', type=int, default=2, help='num of z sampled')
    parser.add_argument('--zdim', type=int, default=128, help='dimension of z')
    parser.add_argument('--load_de', action='store_true', help='whether to load pretrained decoder')
    parser.add_argument('--recon_type', type=str, default='mask', help="which part of decoder get updated")
    parser.add_argument('--resynthesis', action='store_true', help='whether use resynthesis in loss calculation')
    parser.add_argument('--latent_num', type=int, default=1, help='finetune only cvae decoder or both cvae and nvae decoder')
    parser.add_argument('--dlr', type=float, default=8e-5, help='lr for Discrimimator')
    parser.add_argument('--d_step', type=int, default=2, help='update D every some steps')
    args = parser.parse_args()
    cfg = myconf()
    cfg.read(args.cfg_file)




    # state dict paths
    state_dict_paths = os.listdir(args.first_phase_folder)

    clean_encoder_path = [f for f in state_dict_paths if 'clean_encoder' in f]
    clean_encoder_path = args.first_phase_folder + '/' + clean_encoder_path[0]
    noise_encoder_path = [f for f in state_dict_paths if 'noise_encoder' in f]
    noise_encoder_path = args.first_phase_folder + '/' + noise_encoder_path[0]
    clean_decoder_path = [f for f in state_dict_paths if 'clean_decoder' in f]
    clean_decoder_path = args.first_phase_folder + '/' + clean_decoder_path[0]
    noise_decoder_path = [f for f in state_dict_paths if 'noise_decoder' in f]
    noise_decoder_path = args.first_phase_folder + '/' + noise_decoder_path[0]
    noisy_encoder_path = [f for f in state_dict_paths if 'noisy_encoder' in f]
    noisy_encoder_path = args.first_phase_folder + '/' + noisy_encoder_path[0]

    best_clean_encoder_state_dict = torch.load(clean_encoder_path, map_location=device)
    best_noise_encoder_state_dict = torch.load(noise_encoder_path, map_location=device)
    best_clean_decoder_state_dict = torch.load(clean_decoder_path, map_location=device)
    best_noise_decoder_state_dict = torch.load(noise_decoder_path, map_location=device)
    best_noisy_encoder_state_dict = torch.load(noisy_encoder_path, map_location=device) 


    # extract pretrain setups (skipc, recon_type, skipuse)
    cfg_phase1 = myconf()
    cfg_file_phase1 = args.first_phase_folder + '/config.ini' 
    cfg_phase1.read(cfg_file_phase1)
    pretrain_path = cfg_phase1.get("User","pre_clean_encoder")
    setups = pretrain_path.split('/')[-2]
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
        skipuse_str = '012345'
    if 'causal' not in setups:
        causal = False
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
            skipuse_str = ''
            for n in tmp:
                skipuse.append(int(n))
                skipuse_str = skipuse_str + n
        elif 'causal=' in s:
            causal_tmp= s.split('=')[-1]
            causal = (causal_tmp.lower() == 'true')
        elif 'spadd' in s:
            spadd_tmp = s.split('=')[-1]
            spadd = (spadd_tmp.lower() == 'true')

    # extract nsvae setups (w_resi, latentnum, nsvae_model, matching, zdim)
    nsvae_path = args.first_phase_folder.split('/')[-1]
    setups = nsvae_path.split('_')
    zdim = 0
    pre_latent_num = 1
    for s in setups:
        if 'zdim' in s:
            tmp = s.split('=')[-1]
            zdim = int(tmp)
        elif 'latentnum' in s: # determine how many latent vector can be used
            tmp = s.split('=')[-1]
            pre_latent_num = int(tmp)


    log_params = {
        'reload': args.reload,
        'cfg_file': args.cfg_file,
        'model_dir': args.model_dir,
        'first_use_dataset': args.first_use_dataset,
        'first_phase_folder': args.first_phase_folder,
        'causal': args.causal,
        'latent_num': args.latent_num,
        'pre_latent_num': pre_latent_num,
        'decode_update': args.decode_update,
        'num_samples': args.num_samples,
        'zdim': zdim,
        'skipuse_str': skipuse_str,
        'use_sc_phase2': args.use_sc_phase2,
        'load_de': args.load_de,
        'recon_type': args.recon_type,
        'resynthesis': args.resynthesis,
        'dlr': args.dlr,
        'd_step': args.d_step

    }

    # phase 2
    train_nsvae_decoder(cfg, log_params, best_clean_encoder_state_dict, best_clean_decoder_state_dict, best_noise_encoder_state_dict, best_noise_decoder_state_dict, best_noisy_encoder_state_dict, skipuse)


