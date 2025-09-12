#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(main_folder_path)
from utils.read_config import myconf
from utils.eval_metrics import compute_median, compute_mean, EvalMetrics
from model.pvae_module import *
import json
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import random

torch.manual_seed(0)
np.random.seed(0)
def distance(v1, v2, if_logsigma):
    # input time, batch, dim, 2
    if if_logsigma:
        v1 = torch.exp(v1[...,0])
        v2 = torch.exp(v2[...,0])
    res = torch.mean((v1 - v2).pow(2),dim=(0,1)) # dim, 2
    res = torch.sqrt(torch.sum(res))

    return res



def simple_silhouette_score(set1, set2, mean1, mean2, metric):

    if metric == 'euclidean':
        intra_dis_1 = np.sqrt(np.sum((set1 - mean1) ** 2, axis=(1,2)))
        inter_dis_1 = np.sqrt(np.sum((set1 - mean2) ** 2, axis=(1,2)))
        sc_1 = (inter_dis_1 - intra_dis_1) / np.maximum(intra_dis_1, inter_dis_1)
        intra_dis_2 = np.sqrt(np.sum((set2 - mean2) ** 2, axis=(1,2)))
        inter_dis_2 = np.sqrt(np.sum((set2 - mean1) ** 2, axis=(1,2)))
        sc_2 = (inter_dis_2 - intra_dis_2) / np.maximum(intra_dis_2, inter_dis_2)
        sc = np.concatenate((sc_1, sc_2))
        mean_sc = np.mean(sc)


    if metric == 'manhattan':
        intra_dis_1 = np.sum(np.abs((set1 - mean1)), axis=1)
        inter_dis_1 = np.sum(np.abs((set1 - mean2)), axis=1)
        sc_1 = (inter_dis_1 - intra_dis_1) / np.maximum(intra_dis_1, inter_dis_1)
        intra_dis_2 = np.sum(np.abs((set2 - mean2)), axis=1)
        inter_dis_2 = np.sum(np.abs((set2 - mean1)), axis=1)
        sc_2 = (inter_dis_2 - intra_dis_2) / np.maximum(intra_dis_2, inter_dis_2)
        sc = np.concatenate((sc_1, sc_2))
        mean_sc = np.mean(sc)



    if metric == 'cosine':
        mean1 = mean1[None,...]
        mean2 = mean2[None,...]
        intra_dis_1 = cosine_distances(set1, mean1)
        inter_dis_1 = cosine_distances(set1, mean2)
        sc_1 = (inter_dis_1 - intra_dis_1) / np.maximum(intra_dis_1, inter_dis_1)
        intra_dis_2 = cosine_distances(set2, mean2)
        inter_dis_2 = cosine_distances(set2, mean1)
        sc_2 = (inter_dis_2 - intra_dis_2) / np.maximum(intra_dis_2, inter_dis_2)
        sc = np.concatenate((sc_1, sc_2))
        mean_sc = np.mean(sc)

    return mean_sc

def cal_kl(miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

    # B,T,H
    miu1_real = miu1[:,:,:,0]
    miu1_imag = miu1[:,:,:,1]

    miu2_real = miu2[:,:,:,0]
    miu2_imag = miu2[:,:,:,1]

    sigma1 = torch.exp(log_sigma1[:,:,:,0])
    sigma2 = torch.exp(log_sigma2[:,:,:,0])

    delta1_real = delta1[:,:,:,0]
    delta1_imag = delta1[:,:,:,1]

    delta2_real = delta2[:,:,:,0]
    delta2_imag = delta2[:,:,:,1]

    # protection
    # keep abs(delta) <= sigma (protection)
    abs_delta1 = torch.sqrt(delta1_real.pow(2) + delta1_imag.pow(2) + 1e-10)
    temp = sigma1 * 0.99 / (abs_delta1 + 1e-10)

    delta1_real = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_real * temp, delta1_real)
    delta1_imag = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_imag * temp, delta1_imag)

    abs_delta1 = delta1_real.pow(2) + delta1_imag.pow(2)

    # protection
    # keep abs(delta) <= sigma (protection)
    abs_delta2 = torch.sqrt(delta2_real.pow(2) + delta2_imag.pow(2) + 1e-10)
    temp = sigma2 * 0.99 / (abs_delta2 + 1e-10)

    delta2_real = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_real * temp, delta2_real)
    delta2_imag = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_imag * temp, delta2_imag)

    abs_delta2 = delta2_real.pow(2) + delta2_imag.pow(2)

    # B,T,H
    log_det_c1 = torch.log(0.25 * (sigma1.pow(2) - abs_delta1) + 1e-10)
    log_det_c2 = torch.log(0.25 * (sigma2.pow(2) - abs_delta2) + 1e-10)

    coeff = 2 / (sigma2.pow(2) - abs_delta2 + 1e-10)


    trace_term = sigma1 * sigma2 - delta2_real * delta1_real - delta2_imag * delta1_imag

    miu_diff_real = miu2_real - miu1_real
    miu_diff_imag = miu2_imag - miu1_imag
    quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

    kl = 0.5 * torch.sum(coeff * (trace_term + quadra_term) + log_det_c2 - log_det_c1, dim=2) - 128








    return kl
def real_and_imag_mask(noise_stft, speech_stft, noisy_stft):
    # input: numsamples, F, T
    noise_stft = torch.view_as_real(noise_stft) # numsamples, F, T, 2
    speech_stft = torch.view_as_real(speech_stft)

    noise_stft = torch.mean(noise_stft, dim=0)
    speech_stft = torch.mean(speech_stft, dim=0)
    noisy_stft = torch.mean(noisy_stft, dim=0)
    real_mask = torch.pow(speech_stft[:,:,0], 2) / (torch.pow(speech_stft[:,:,0], 2) + torch.pow(noise_stft[:,:,0], 2) + 1e-10)
    imag_mask = torch.pow(speech_stft[:,:,1], 2) / (torch.pow(speech_stft[:,:,1], 2) + torch.pow(noise_stft[:,:,1], 2) + 1e-10)

    est_real = real_mask * noisy_stft[:,:,0]
    est_imag = imag_mask * noisy_stft[:,:,1]

    est_stft = torch.complex(est_real, est_imag)

    return est_stft


def complex_mask(noise_stft, speech_stft, noisy_stft):
    # input: numsamples, F, T
    noisy_stft = torch.complex(noisy_stft[:,:,:,0], noisy_stft[:,:,:,1])
    noisy_stft = torch.squeeze(noisy_stft, dim=0)
    noise_stft = torch.mean(noise_stft, dim=0)
    speech_stft = torch.mean(speech_stft, dim=0)

    complex_mask = speech_stft / (speech_stft + noise_stft + 1e-10)
    print(speech_stft[0,0], noise_stft[0,0], speech_stft[0,0] + noise_stft[0,0])
    est_stft = complex_mask * noisy_stft

    return est_stft


def phase_sensitive_mask(noise_stft, speech_stft, noisy_stft):

    speech_stft = torch.mean(speech_stft, dim=0)
    speech_phase = torch.angle(speech_stft)
    speech_mag = torch.abs(speech_stft)
    noise_stft = torch.mean(noise_stft, dim=0)
    noise_mag = torch.abs(noise_stft)
    noisy_stft = torch.complex(noisy_stft[:,:,:,0], noisy_stft[:,:,:,1])
    noisy_stft = torch.squeeze(noisy_stft, dim=0)
    noisy_phase = torch.angle(noisy_stft)
    noisy_mag = torch.abs(noisy_stft)

    diff_angle = speech_phase - noisy_phase
    mask_psm = (speech_mag / (speech_mag + noise_mag + 1e-10)) * torch.cos(diff_angle)

    est_stft = mask_psm *  noisy_mag * torch.exp(1j * speech_phase)

    return est_stft


def run(clean_encoder, clean_decoder, noise_decoder, noisy_encoder, statedict_folder, file_list, label_folder, testset, info_file, eval_metrics, 
        STFT_dict, model_params, mean_file, std_file, 
        resfolder, resjson, device, save_output):
    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_pesq_wb = []
    list_pesq_nb = []
    list_estoi = []
    list_dis_mean = []
    list_dis_sigma = []
    list_dis_delta = []

    data_check_dir = resfolder
    if not os.path.exists(data_check_dir):
        os.makedirs(data_check_dir)

    test_noisy_speech_dict = {}

    if info_file != None:
        if testset == 'demand':
            snr_dict = {}
            with open(info_file, 'r') as file:
                for line in file:
                    # Split the line into file name and SNR
                    file_name, noise, snr = line.strip().split()  # Adjust delimiter if necessary (e.g., if comma-separated, use .split(','))
                    snr_dict[file_name] = float(snr)  # Store SNR as a float for numerical calculations
    z_dim = model_params["zdim"]
    clean_z_set = np.empty((0,z_dim, 2))
    clean_miu_z_set = np.empty((0,z_dim, 2))    
    noise_z_set = np.empty((0,z_dim, 2))
    noise_miu_z_set = np.empty((0,z_dim, 2))   

    for audio_file in tqdm(file_list):
                

        nfft = STFT_dict['nfft']
        hop = STFT_dict['hop']
        wlen = STFT_dict['wlen']
        trim = STFT_dict['trim']
        window = torch.hann_window(wlen)
        window = window.to(device)

        x, fs_x = sf.read(audio_file)
        if fs_x != 16000:
            audio_resampled = librosa.resample(x, orig_sr=fs_x, target_sr=16000)
            x =audio_resampled   

        # label file
        if testset == 'dns2021' or testset == 'dns2021_official' or testset == 'lowsnr_dns':
            filefullname = audio_file.split('.')[0]
            try:
                snr = filefullname.split('_')[-4]
                snr = int(snr[3:])
            except:
                if 'clean' in filefullname:
                    snr = 100
                else:
                    snr = -100
            fileid = filefullname.split('_')[-1]
            clean_filename = label_folder + '/' + 'clean_fileid_' + fileid + '.wav'
            clean_x, fs_x = sf.read(clean_filename)
            if fs_x != 16000:
                audio_resampled = librosa.resample(clean_x, orig_sr=fs_x, target_sr=16000)
                clean_x =audio_resampled 

            filename = 'noisy_fileid_' + fileid + '_' + str(snr)
            p_id = filefullname.split('/')[-1]
            # print(filename)
        elif testset == 'wsj0' or testset == 'lowsnr_wsj':
            filefullname = audio_file.split('/')[-1]
            filefullname = filefullname.split('.')[0]
            snr = filefullname.split('_')[-1]
            snr = int(snr)
            clean_name = filefullname.split('_')[0]
            fileid = clean_name
            clean_filename = label_folder + '/' + clean_name + '.wav'
            clean_x, _ = sf.read(clean_filename)
            filename = filefullname
            p_id = filefullname.split('/')[-1]

        elif testset == 'demand':
            filefullname = audio_file.split('/')[-1]
            filefullname = filefullname.split('.')[0]
            fileid = filefullname
            clean_filename = label_folder + '/' + filefullname + '.wav'
            clean_x, fs = sf.read(clean_filename)
            if fs != fs_x:
                audio_resampled = librosa.resample(clean_x, orig_sr=fs, target_sr=fs_x)
                clean_x =audio_resampled
            filename = filefullname
            p_id = filefullname
            snr = snr_dict[filefullname]


        
        #####################
        # preprocess the input
        #####################
        tmp_x = torch.from_numpy(x)
        tmp_x = tmp_x.float()
        tmp_x = tmp_x.to(device)
        tmp_x = tmp_x[None, ...]
        bs, time_len = tmp_x.shape

        tmp_cleanx = torch.from_numpy(clean_x)
        tmp_cleanx = tmp_cleanx.float()
        tmp_cleanx = tmp_cleanx.to(device)
        tmp_cleanx = tmp_cleanx[None, ...]


        with torch.no_grad():
            (z_speech_noisy, miu_speech_noisy, log_sigma_speech_noisy, delta_speech_noisy, 
            z_noise_noisy, miu_noise_noisy, log_sigma_noise_noisy, delta_noise_noisy, 
            skiper_noisy, C_noisy, F_noisy, stft_x_noisy) = noisy_encoder(tmp_x, train=False) 

            if latent_num == 2:
                tmp = z_speech_noisy.view(z_speech_noisy.shape[0]*z_speech_noisy.shape[1],z_speech_noisy.shape[2],2)
                randint = random.sample(range(0, z_speech_noisy.shape[0]*z_speech_noisy.shape[1]), 40)
                tmp = tmp[randint,:,:]
                np_z_clean = tmp.cpu().detach().numpy()
                clean_z_set = np.concatenate((clean_z_set, np_z_clean), axis=0)

                tmp = z_noise_noisy.view(z_noise_noisy.shape[0]*z_noise_noisy.shape[1],z_noise_noisy.shape[2],2)
                randint = random.sample(range(0, z_noise_noisy.shape[0]*z_noise_noisy.shape[1]), 40)
                tmp = tmp[randint,:,:]
                np_z_noise = tmp.cpu().detach().numpy()
                noise_z_set = np.concatenate((noise_z_set, np_z_noise), axis=0)

                tmp = miu_speech_noisy.view(miu_speech_noisy.shape[0]*miu_speech_noisy.shape[1],miu_speech_noisy.shape[2],2)
                randint = random.sample(range(0, miu_speech_noisy.shape[0]*miu_speech_noisy.shape[1]), 50)
                tmp = tmp[randint,:,:]
                np_z_clean = tmp.cpu().detach().numpy()
                clean_miu_z_set = np.concatenate((clean_miu_z_set, np_z_clean), axis=0)

                tmp = miu_noise_noisy.view(miu_noise_noisy.shape[0]*miu_noise_noisy.shape[1],miu_noise_noisy.shape[2],2)
                randint = random.sample(range(0, miu_noise_noisy.shape[0]*miu_noise_noisy.shape[1]), 50)
                tmp = tmp[randint,:,:]
                np_z_noise = tmp.cpu().detach().numpy()
                noise_miu_z_set = np.concatenate((noise_miu_z_set, np_z_noise), axis=0)

            if model_params["latent_to_use"] == 1:
                skiper_clean = []
                if model_params['nsvae_model'] == 'double':
                    for idx, skip in enumerate(skiper_noisy):
                        C = skip.shape[1]
                        skiper_clean.append(skip[:,:int(C//2), :, :, :])
                elif model_params['nsvae_model'] == 'original':
                    skiper_clean = skiper_noisy
                elif model_params['nsvae_model'] == 'adapt':
                    for idx, skip in enumerate(skiper_noisy):
                        if (len(skiper_noisy) - 1 - idx) in model_params['skipuse']:
                            C = skip.shape[1]
                            skiper_clean.append(skip[:,:int(C//2), :, :, :])
                        else:
                            skiper_clean.append(skip)
                C = skiper_clean[-1].shape[1]
                clean_z, miu_clean, log_sigma_clean, delta_clean, _, _, _, stft_x_clean = clean_encoder(tmp_cleanx, train=False)
                recon_sig_clean, predict_stft_clean = clean_decoder(stft_x_noisy, z_speech_noisy, skiper_clean, C, F_noisy, train=False) # B * num_samples, time len  

                recon_sig = torch.mean(recon_sig_clean,dim=0)
                recon_res = recon_sig.cpu().detach().numpy()
                recon_noise = None


            if model_params["latent_to_use"] == 2:
                skiper_clean = []
                skiper_noise = []
                if model_params['nsvae_model'] == 'double':
                    for idx, skip in enumerate(skiper_noisy):
                        C = skip.shape[1]
                        skiper_clean.append(skip[:,:int(C//2), :, :, :])
                        skiper_noise.append(skip[:,int(C//2):, :, :, :])
                elif model_params['nsvae_model'] == 'original':
                    skiper_clean = skiper_noisy
                    skiper_noise = skiper_noisy
                elif model_params['nsvae_model'] == 'adapt':
                    for idx, skip in enumerate(skiper_noisy):
                        if (len(skiper_noisy) - 1 - idx) in model_params['skipuse']:
                            C = skip.shape[1]
                            skiper_clean.append(skip[:,:int(C//2), :, :, :])
                            skiper_noise.append(skip[:,int(C//2):, :, :, :])
                        else:
                            skiper_clean.append(skip)
                            skiper_noise.append(skip)
                C = skiper_clean[-1].shape[1]
                clean_z, miu_clean, log_sigma_clean, delta_clean, _, _, _, stft_x_clean = clean_encoder(tmp_cleanx, train=False)
                # z_noise_noisy, miu, log_sigma, delta, skiper, C, F, stft_x_noisy = noise_encoder(tmpnoise_x, train=False)            
                recon_sig_clean, predict_stft_clean = clean_decoder(stft_x_noisy, z_speech_noisy, skiper_clean, C, F_noisy, train=False) # B * num_samples, time len                            
                recon_sig_noise, predict_stft_noise = noise_decoder(stft_x_noisy, z_noise_noisy, skiper_noise, C, F_noisy, train=False) # B * num_samples, time len 
                clean_est = torch.mean(recon_sig_clean,dim=0)
                noise_est = torch.mean(recon_sig_noise,dim=0)
                clean_est = clean_est.cpu().detach().numpy()
                noise_est = noise_est.cpu().detach().numpy()
                # clean time signal as estimate
                if model_params['outtype'] == 'clean_direct':
                    recon_sig = torch.mean(recon_sig_clean,dim=0)
                    # recon_sig = recon_sig_clean[0]
                    recon_res = recon_sig.cpu().detach().numpy()

                    # recon_noise = torch.mean(recon_sig_noise,dim=0)
                    recon_noise = None              
                # real and imag mask
                elif model_params['outtype'] == 'real_imag_mask':
                    est_stft = real_and_imag_mask(predict_stft_noise, predict_stft_clean, stft_x_noisy)
                    recon_res = torch.istft(est_stft, n_fft=nfft, hop_length=hop, win_length=wlen, window=window,return_complex=False)
                    recon_res = recon_res.cpu().detach().numpy()
                    recon_noise = None
                # complex mask
                elif model_params['outtype'] == 'complex_mask':
                    est_stft = complex_mask(predict_stft_noise, predict_stft_clean, stft_x_noisy)
                    recon_res = torch.istft(est_stft, n_fft=nfft, hop_length=hop, win_length=wlen, window=window,return_complex=False)
                    recon_res = recon_res.cpu().detach().numpy()
                    recon_noise = None

                # phase sensitive mask
                elif model_params['outtype'] == 'phase_mask':
                    est_stft = phase_sensitive_mask(predict_stft_noise, predict_stft_clean, stft_x_noisy)
                    recon_res = torch.istft(est_stft, n_fft=nfft, hop_length=hop, win_length=wlen, window=window,return_complex=False)
                    recon_res = recon_res.cpu().detach().numpy()
                    recon_noise = None


            # distance from clean to noisy
            dis_mean = distance(miu_clean, miu_speech_noisy, False)
            dis_sigma = distance(log_sigma_clean, log_sigma_speech_noisy, True)
            dis_delta = distance(delta_clean, delta_speech_noisy, False)






        if eval_metrics.metric == 'no':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = 0, 0, 0, 0, 0, 0
        elif eval_metrics.metric == 'all':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=recon_res, x_ref=clean_x, fs=fs_x, name=audio_file)
        else:
            metric_res = eval_metrics.eval(x_est=recon_res, x_ref=x, fs=fs_x)
            if eval_metrics.metric == 'sisdr':
                sisdr = metric_res
                rmse, pesq, pesq_wb, pesq_nb, estoi = 0,0,0,0,0
            if eval_metrics.metric == 'rmse':
                rmse = metric_res
                sisdr, pesq, pesq_wb, pesq_nb, estoi = 0,0,0,0,0


        list_rmse.append(rmse)
        list_sisdr.append(sisdr)
        list_pesq.append(pesq)
        list_pesq_wb.append(pesq_wb)
        list_pesq_nb.append(pesq_nb)
        list_estoi.append(estoi)
        list_dis_mean.append(dis_mean.item())
        list_dis_sigma.append(dis_sigma.item())
        list_dis_delta.append(dis_delta.item())

        if save_output:
            try:
                sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_clean.wav", clean_est, fs_x)
                sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_noise.wav", noise_est, fs_x)
            except:
                pass
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_noisy.wav", x, fs_x)
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_enhanced.wav", recon_res, fs_x)
        
        test_noisy_speech_dict[fileid] = {
            'p_id': p_id,
            'utt_name': fileid,
            'snr': snr,
            'sisdr': sisdr,
            'pesq': pesq,
            'pesq_wb': pesq_wb,
            'pesq_nb': pesq_nb,
            'estoi': estoi,
            'dis_mean': dis_mean.item(),
            'dis_sigma': dis_sigma.item(),
            'dis_delta': dis_delta.item()
        }

    np_rmse = np.array(list_rmse)
    np_sisdr = np.array(list_sisdr)
    np_pesq = np.array(list_pesq)
    np_pesq_wb = np.array(list_pesq_wb)
    np_pesq_nb = np.array(list_pesq_nb)
    np_estoi = np.array(list_estoi)
    np_dis_mean = np.array(list_dis_mean)
    np_dis_sigma = np.array(list_dis_sigma)
    np_dis_delta = np.array(list_dis_delta)

    if latent_num == 2:
        mean_z_speech = np.mean(clean_z_set, axis=0, keepdims=True)
        mean_z_noise = np.mean(noise_z_set, axis=0, keepdims=True)

        mean_sc = simple_silhouette_score(clean_z_set, noise_z_set, mean_z_speech, mean_z_noise, 'euclidean')
        
        var_speech = np.var(clean_z_set, axis=0)
        var_avg_speech = np.mean(var_speech, axis=0)

        var_noise = np.var(noise_z_set, axis=0)
        var_avg_noise = np.mean(var_noise, axis=0)

        var_speech_zmiu = np.var(clean_miu_z_set, axis=0)
        var_avg_speech_zmiu = np.mean(var_speech_zmiu, axis=0)

        var_noise_zmiu = np.var(noise_miu_z_set, axis=0)
        var_avg_noise_zmiu = np.mean(var_noise_zmiu, axis=0)

        mean_dis = np.sum((mean_z_speech-mean_z_noise)**2, axis=(1,2))

        mean_sc = simple_silhouette_score(clean_z_set, noise_z_set, mean_z_speech, mean_z_noise, "euclidean")

    file_path = resjson

    # Write the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(test_noisy_speech_dict, json_file, indent=4)

    rmse_mean, rmse_interval = compute_mean(np_rmse)
    sisdr_mean, sisdr_interval = compute_mean(np_sisdr)
    pesq_mean, pesq_interval = compute_mean(np_pesq)
    pesq_wb_mean, pesq_wb_interval = compute_mean(np_pesq_wb)
    pesq_nb_mean, pesq_nb_interval = compute_mean(np_pesq_nb)
    estoi_mean, estoi_interval = compute_mean(np_estoi)
    meandis_mean, meandis_interval = compute_mean(np_dis_mean)
    sigmadis_mean, sigmadis_interval = compute_mean(np_dis_sigma)
    deltadis_mean, deltadis_interval = compute_mean(np_dis_delta)
    # kldiv_mean, kldiv_interval = compute_mean(np_kldiv)

    rmse_median, rmse_ci = compute_median(np_rmse)
    sisdr_median, sisdr_ci = compute_median(np_sisdr)
    pesq_median, pesq_ci = compute_median(np_pesq)
    pesq_wb_median, pesq_wb_ci = compute_median(np_pesq_wb)
    pesq_nb_median, pesq_nb_ci = compute_median(np_pesq_nb)
    estoi_median, estoi_ci = compute_median(np_estoi)

    with open(data_check_dir+'log.txt', 'w') as f:
        print('Re-synthesis finished', file=f)

        print('state dict folder: {}'.format(statedict_folder), file=f) 
        print("mean evaluation", file=f)
        print('mean rmse score: {:.4f} +/- {:.4f}'.format(rmse_mean, rmse_interval), file=f)
        print('mean sisdr score: {:.1f} +/- {:.1f}'.format(sisdr_mean, sisdr_interval), file=f)
        print('mean pypesq score: {:.2f} +/- {:.2f}'.format(pesq_mean, pesq_interval), file=f)
        print('mean pesq wb score: {:.2f} +/- {:.2f}'.format(pesq_wb_mean, pesq_wb_interval), file=f)
        print('mean pesq nb score: {:.2f} +/- {:.2f}'.format(pesq_nb_mean, pesq_nb_interval), file=f)
        print('mean estoi score: {:.2f} +/- {:.2f}'.format(estoi_mean, estoi_interval), file=f)
        # print('mean kldiv score: {:.2f} +/- {:.2f}'.format(kldiv_mean, kldiv_interval), file=f)
        if latent_num == 2:
            print("mean sc is {:.4f}".format(mean_sc), file=f)
            print("mean speech noise dis is {:.4f}".format(mean_dis.item()), file=f)
            # print("mean speech", file=f)
            # print(mean_z_speech, file=f)
            print("var avg speech", file=f)
            print(var_avg_speech, file=f)
            print("var avg speech_zmiu", file=f)
            print(var_avg_speech_zmiu, file=f)
            # print("mean noise", file=f)
            # print(mean_z_noise, file=f)
            print("var avg noise", file=f)
            print(var_avg_noise, file=f)
            print("var avg noise zmiu", file=f)
            print(var_avg_noise_zmiu, file=f)

        print("dis to clean evaluation", file=f)
        print('dis mean score: {:.4f} +/- {:.4f}'.format(meandis_mean, meandis_interval), file=f)
        print('dis sigma  score: {:.4f} +/- {:.4f}'.format(sigmadis_mean, sigmadis_interval), file=f)
        print('dis delta score: {:.4f} +/- {:.4f}'.format(deltadis_mean, deltadis_interval), file=f)


        print("Median evaluation", file=f)
        print('median rmse score: {:.4f} +/- {:.4f}'.format(rmse_median, rmse_ci), file=f)
        print('median sisdr score: {:.1f} +/- {:.1f}'.format(sisdr_median, sisdr_ci), file=f)
        print('median pypesq score: {:.2f} +/- {:.2f}'.format(pesq_median, pesq_ci), file=f)
        print('median pesq wb score: {:.2f} +/- {:.2f}'.format(pesq_wb_median, pesq_wb_ci), file=f)
        print('median pesq nb score: {:.2f} +/- {:.2f}'.format(pesq_nb_median, pesq_nb_ci), file=f)
        print('median estoi score: {:.2f} +/- {:.2f}'.format(estoi_median, estoi_ci), file=f)

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict_folder', type=str, default=None, help='folder of trained models of nsvae')
    # parser.add_argument('--label_folder', type=str, default=None, help='the folder of clean speech files')
    # parser.add_argument('--state_dict_decoder', type=str, default=None, help='pretrained decoder model state')
    parser.add_argument('--testset', type=str, default='dns2021_official', choices=['dns2021', 'wsj0', 'demand', 'dns2021_official', 'lowsnr_wsj', 'lowsnr_dns'], help='test on wsj or voicebank')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu')
    parser.add_argument('--resfolder', type=str, default='testres/', help='the folder to save results')
    parser.add_argument('--metric', type=str, default='all', help='the metric to calculate')
    parser.add_argument('--num_samples', type=int, default=1, help='num of samples sampled from nsvae')
    parser.add_argument('--latent_to_use', type=int, default=1, help='num of samples sampled from nsvae')
    parser.add_argument('--outtype', type=str, default='clean_direct', help='the method to obtain clean speech')
    parser.add_argument('--resjson', type=str, default='res.json', help='the json file to save evaluation results')
    # parser.add_argument('--latent_check', action='store_true', help='whether to check latent space')
    parser.add_argument('--save_output', action='store_true', help='whether to save output files')
    args = parser.parse_args()

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    eval_metrics = EvalMetrics(metric=args.metric)

    models_path = args.state_dict_folder
    models_path = models_path.split('/')[-1]
    resfolder = args.resfolder + args.testset +  "_" + args.outtype + '_' + models_path + '/'
    resjson = resfolder + 'eval.json'


    # File path config
    if args.testset == 'dns2021_official':
        file_list = librosa.util.find_files('/data/data/DNS-Challenge/test_set/synthetic/no_reverb/noisy', ext='wav')
        # file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/noise_test_mix_short', ext='wav')
        label_folder = '/data/data/DNS-Challenge/test_set/synthetic/no_reverb/clean_old_naming'
        info_file = None
        # label_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/clean_test_mix', ext='wav')
        # file_list = librosa.util.find_files('/data2/jiatong_data/WSJ0/val_debug', ext='wav')
    elif args.testset == 'dns2021':
        # file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/really_small_noisydata', ext='wav')
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/noisy_test_mix_short', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/dns/spk_split_dataset/clean_test_mix_short'
        info_file = None
        # label_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/clean_test_mix', ext='wav')
        # file_list = librosa.util.find_files('/data2/jiatong_data/WSJ0/val_debug', ext='wav')
    elif args.testset == 'lowsnr_dns':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/data_20h_lowsnr/test/noisy', ext='wav')
        # file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/noise_test_mix_short', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/dns/spk_split_dataset/data_20h_lowsnr/test/clean'
        info_file = None  

    elif args.testset == 'wsj0':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/WSJ0/WSJ0_QUT', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/WSJ0/test_si_et_05'
        info_file = None
    elif args.testset == 'lowsnr_wsj':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/WSJ0/WSJ0_QUT_lowsnr_-20_0', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/WSJ0/test_si_et_05'
        info_file = None
    elif args.testset == 'lowsnr_wsj':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/WSJ0/WSJ0_QUT_lowsnr_-20_0', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/WSJ0/test_si_et_05'
        info_file = None
    elif args.testset == 'demand':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/voicebank_demand/noisy_testset_wav_16k/noisy_testset_wav_16k', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/voicebank_demand/clean_testset_wav_16k'
        info_file = '/data1/corpora/jiatong_data/voicebank_demand/log_testset.txt'

    print(f'Test on {args.testset}, totl audio files {len(file_list)}')

    # load DVAE model
    model_folder = args.state_dict_folder
    cfg_file = os.path.join(model_folder, 'config.ini')
    cfg = myconf()
    cfg.read(cfg_file)



    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')

    trim = cfg.getboolean('STFT', 'trim')

    # state dict paths
    state_dict_paths = os.listdir(args.state_dict_folder)
    clean_encoder_path = [f for f in state_dict_paths if 'clean_encoder' in f]
    clean_encoder_path = args.state_dict_folder + '/' + clean_encoder_path[0]
    # noise_encoder_path = [f for f in state_dict_paths if 'noise_encoder' in f]
    # noise_encoder_path = args.state_dict_folder + '/' + noise_encoder_path[0]


    clean_decoder_path = [f for f in state_dict_paths if 'clean_decoder' in f]
    clean_decoder_path = args.state_dict_folder + '/' + clean_decoder_path[0]
    noise_decoder_path = [f for f in state_dict_paths if 'noise_decoder' in f]
    noise_decoder_path = args.state_dict_folder + '/' + noise_decoder_path[0]
    noisy_encoder_path = [f for f in state_dict_paths if 'noisy_encoder' in f]
    noisy_encoder_path = args.state_dict_folder + '/' + noisy_encoder_path[0]

    # extract pretrain setups (skipc, recon_type, skipuse)
    pretrain_path = cfg.get("User","pre_clean_encoder")
    setups = pretrain_path.split('/')[-2]
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
    if 'causal' not in setups:
        causal = False
    if 'spadd' not in setups:
        spadd = False
    if 'fcl=' not in setups:
        fcl = False
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
        elif 'causal=' in s:
            causal_tmp= s.split('=')[-1]
            causal = (causal_tmp.lower() == 'true')
        elif 'spadd' in s:
            spadd_tmp = s.split('=')[-1]
            spadd = (spadd_tmp.lower() == 'true')
        elif 'fcl=' in s:
            fcl = s.split('=')[-1]
            fcl = (fcl.lower() == 'true')


    # extract nsvae setups (w_resi, latentnum, nsvae_model, matching, zdim)
    nsvae_path = args.state_dict_folder.split('/')[-1]
    setups = nsvae_path.split('_')
    zdim = 0
    w_resi = 0
    nsvae_model = 'original'
    latent_num = 1
    matching = 'speech'
    for s in setups:
        if 'zdim' in s:
            tmp = s.split('=')[-1]
            zdim = int(tmp)
        elif 'wresi' in s: # determine whether use skipc to decoder
            tmp = s.split('=')[-1]
            w_resi = float(tmp)
        elif 'nsvae=' in s: # determine which nsvae model to use
            tmp = s.split('=')[-1]
            nsvae_model = tmp
        elif 'latentnum' in s: # determine how many latent vector can be used
            tmp = s.split('=')[-1]
            latent_num = int(tmp)
        elif 'match' in s: # determine which skipc to match
            tmp = s.split('=')[-1] 
            matching = tmp

    # load model
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()

    if skipc == 'False':
        if not fcl:
            if not spadd:
                clean_model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
                # noise_model_encoder = pvae_dccrn_encoder_no_skip(net_params, device, zdim, nfft, hop, wlen, args.num_samples)
                clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
                noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params,causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
            else:
                clean_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
                clean_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)
                # noise_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)          
        else:
            if not spadd:
                clean_model_encoder = pvae_dccrn_encoder_no_skip_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
                # noise_model_encoder = pvae_dccrn_encoder_no_skip(net_params, device, zdim, nfft, hop, wlen, args.num_samples)
                clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
                noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params,causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
            else:
                clean_model_encoder = pvae_dccrn_encoder_skip_prepare_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
                clean_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)
                # noise_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, num_samples)
                noise_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type, skipuse)            


    else:
        clean_model_encoder = pvae_dccrn_encoder(net_params,causal, device, zdim, nfft, hop, wlen, args.num_samples)
        # noise_model_encoder = pvae_dccrn_encoder(net_params, device, zdim, nfft, hop, wlen, args.num_samples)
        clean_model_decoder = pvae_dccrn_decoder(net_params,causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse)            
        noise_model_decoder = pvae_dccrn_decoder(net_params,causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse) 
    if not spadd:
        if nsvae_model == 'original':
            if not fcl:
                noisy_model_encoder = nsvae_dccrn_encoder_original(net_params,causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)
            else:
                noisy_model_encoder = nsvae_dccrn_encoder_original_fc_latent(net_params,causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)
        elif nsvae_model == 'double':
            noisy_model_encoder = nsvae_dccrn_encoder_double_channel(net_params,causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)
        elif nsvae_model == 'adapt':
            noisy_model_encoder = nsvae_dccrn_encoder_adapt_channel(net_params,causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num, skipuse) 
    else:
        if not fcl:
            noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)
        else:
            noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase_fc_latent(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)            
    clean_model_encoder.load_state_dict(torch.load(clean_encoder_path, map_location=device))
    # noise_model_encoder.load_state_dict(torch.load(noise_encoder_path, map_location=device))
    clean_model_decoder.load_state_dict(torch.load(clean_decoder_path, map_location=device))
    # noise_model_decoder.load_state_dict(torch.load(noise_decoder_path, map_location=device))
    noisy_model_encoder.load_state_dict(torch.load(noisy_encoder_path, map_location=device))
    clean_model_encoder = clean_model_encoder.to(device)
    # noise_model_encoder = noise_model_encoder.to(device)
    clean_model_decoder = clean_model_decoder.to(device)
    # noise_model_decoder = noise_model_decoder.to(device)
    noisy_model_encoder = noisy_model_encoder.to(device)
    clean_model_encoder.eval()
    # noise_model_encoder.eval()
    clean_model_decoder.eval()
    # noise_model_decoder.eval()
    noisy_model_encoder.eval()
    # print('encoder params: %.2fM' % (sum(p.numel() for p in model_encoder.parameters()) / 1e6))
    # print('decoder params: %.2fM' % (sum(p.numel() for p in model_decoder.parameters()) / 1e6))

    # load mean and std
    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')

    model_params = {
        'causal': causal,
        "skipc": skipc,
        "skipuse": skipuse,
        "recontype": recon_type,
        "w_resi": w_resi,
        "latentnum": latent_num,
        "nsvae_model": nsvae_model,
        "matching": matching,
        "zdim": zdim,
        "num_samples": args.num_samples,
        "outtype": args.outtype,
        "latent_to_use": args.latent_to_use
    }

    STFT_dict = {}
    STFT_dict['nfft'] = nfft
    STFT_dict['hop'] = hop
    STFT_dict['wlen'] = wlen
    STFT_dict['fs'] = fs
    STFT_dict['trim'] = trim

    print('='*80)
    print('STFT params')
    print(f'fs: {fs}')
    print(f'wlen: {wlen}')
    print(f'hop: {hop}')
    print(f'nfft: {nfft}')
    print('='*80)

    run(clean_model_encoder, clean_model_decoder, noise_model_decoder, noisy_model_encoder, args.state_dict_folder, file_list, label_folder, args.testset, info_file,
        eval_metrics, STFT_dict, model_params, mean_file, std_file, resfolder, resjson, device, args.save_output)
