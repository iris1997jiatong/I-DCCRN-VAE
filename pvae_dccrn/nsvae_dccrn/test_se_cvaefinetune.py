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

torch.manual_seed(0)
np.random.seed(0)
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


def run(clean_encoder, clean_decoder, noise_decoder, noisy_encoder, noisy_clean_decoder, noisy_noise_decoder,
         statedict_folder, file_list, label_folder, testset, info_file, eval_metrics, 
        STFT_dict, model_params, mean_file, std_file, 
        resfolder, resjson, device, save_output):
    
    print(resfolder)
    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_pesq_wb = []
    list_pesq_nb = []
    list_estoi = []


    list_delta_rmse = []
    list_delta_sisdr = []
    list_delta_pesq = []
    list_delta_pesq_wb = []
    list_delta_pesq_nb = []
    list_delta_estoi = []

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
    clean_z_set = np.empty((0,z_dim))
    clean_miuz_set = np.empty((0,z_dim))
    noise_z_set = np.empty((0,z_dim)) 
    noise_miuz_set = np.empty((0,z_dim)) 

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
            # if trim:
            #     clean_x, _ = librosa.effects.trim(clean_x, top_db=30)

            filename = 'noisy_fileid_' + fileid + '_' + str(snr)
            p_id = filefullname.split('/')[-1]
            # print(filename)
        elif testset == 'wsj0' or testset == 'lowsnr_wsj':
            filefullname = audio_file.split('/')[-1]
            filefullname = filefullname.split('.')[0]
            # print(audio_file)
            # print(filefullname)
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

        # x = np.tile(x, 5)
        # clean_x = np.tile(clean_x, 5)

        
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

            (z_noisy_speech, miu_noisy_speech, log_sigma_noisy_speech, delta_noisy_speech, 
             z_noisy_noise, miu_noisy_noise, log_sigma_noisy_noise, delta_noisy_noise, skiper_noisy, 
             C, F, stft_x_noisy) = noisy_encoder(tmp_x, train=False)
            # z_noisy_speech, miu, log_sigma, delta, skiper_noisy, C, F, stft_x_noisy = clean_encoder(tmp_cleanx, train=False)
            # z_noisy_noise = None
            if model_params['phase'] == 1:
                if model_params['latent_to_use'] == 1:
                    recon_sig_clean, predict_stft_clean = clean_decoder(stft_x_noisy, z_noisy_speech, skiper_noisy, C, F, train=False)
                    recon_sig_clean = torch.mean(recon_sig_clean,dim=0)
                    recon_res = recon_sig_clean.cpu().detach().numpy()
                    recon_noise = None
                elif model_params['latent_to_use'] == 2:
                    recon_sig_clean, predict_stft_clean = clean_decoder(stft_x_noisy, z_noisy_speech, skiper_noisy, C, F, train=False)
                    recon_sig_noise, predict_stft_noise = noise_decoder(stft_x_noisy, z_noisy_noise, skiper_noisy, C, F, train=False)
                    clean_est = torch.mean(recon_sig_clean, dim=0)
                    noise_est = torch.mean(recon_sig_noise, dim=0)
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
            elif model_params['phase'] == 2:
                if model_params['latent_to_use'] == 1:
                    recon_sig_clean, predict_stft_clean = noisy_clean_decoder(stft_x_noisy, z_noisy_speech, skiper_noisy, C, F, train=False, pad='sig')
                    recon_sig_clean = torch.mean(recon_sig_clean,dim=0)
                    recon_res = recon_sig_clean.cpu().detach().numpy()
                    recon_noise = None
                elif model_params['latent_to_use'] == 2:
                    recon_sig_clean, predict_stft_clean = noisy_clean_decoder(stft_x_noisy, z_noisy_speech, skiper_noisy, C, F, train=False, pad='sig')
                    recon_sig_noise, predict_stft_noise = noisy_noise_decoder(stft_x_noisy, z_noisy_noise, skiper_noisy, C, F, train=False, pad='sig')
                    clean_est = torch.mean(recon_sig_clean, dim=0)
                    noise_est = torch.mean(recon_sig_noise, dim=0)
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



        if eval_metrics.metric == 'no':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = 0, 0, 0, 0, 0, 0
        elif eval_metrics.metric == 'all':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=recon_res, x_ref=clean_x, fs=fs_x, name=audio_file)
            rmse_noisy, sisdr_noisy, pesq_noisy, pesq_wb_noisy, pesq_nb_noisy, estoi_noisy = eval_metrics.eval(x_est=x, x_ref=clean_x, fs=fs_x, name=audio_file)
        else:
            metric_res = eval_metrics.eval(x_est=recon_res, x_ref=x, fs=fs_x)
            if eval_metrics.metric == 'sisdr':
                sisdr = metric_res
                rmse, pesq, pesq_wb, pesq_nb, estoi = 0,0,0,0,0
            if eval_metrics.metric == 'rmse':
                rmse = metric_res
                sisdr, pesq, pesq_wb, pesq_nb, estoi = 0,0,0,0,0


        list_rmse.append(rmse)
        list_delta_rmse.append(rmse - rmse_noisy)
        list_sisdr.append(sisdr)
        list_delta_sisdr.append(sisdr - sisdr_noisy)
        list_pesq.append(pesq)
        list_delta_pesq.append(pesq - pesq_noisy)
        list_pesq_wb.append(pesq_wb)
        list_delta_pesq_wb.append(pesq_wb - pesq_wb_noisy)
        list_pesq_nb.append(pesq_nb)
        list_delta_pesq_nb.append(pesq_nb - pesq_nb_noisy)
        list_estoi.append(estoi)
        list_delta_estoi.append(estoi - estoi_noisy)

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
            'delta_sisdr': sisdr - sisdr_noisy,
            'pesq': pesq,
            'delta_pesq': pesq - pesq_noisy,
            'pesq_wb': pesq_wb,
            'delta_pesq_wb': pesq_wb - pesq_wb_noisy,
            'pesq_nb': pesq_nb,
            'delta_pesq_nb': pesq_nb - pesq_nb_noisy,
            'estoi': estoi,
            'delta_estoi': estoi - estoi_noisy
        }

    np_rmse = np.array(list_rmse)
    np_delta_rmse = np.array(list_delta_rmse)
    np_sisdr = np.array(list_sisdr)
    np_delta_sisdr = np.array(list_delta_sisdr)
    np_pesq = np.array(list_pesq)
    np_delta_pesq = np.array(list_delta_pesq)
    np_pesq_wb = np.array(list_pesq_wb)
    np_delta_pesq_wb = np.array(list_delta_pesq_wb)
    np_pesq_nb = np.array(list_pesq_nb)
    np_delta_pesq_nb = np.array(list_delta_pesq_nb)
    np_estoi = np.array(list_estoi)
    np_delta_estoi = np.array(list_delta_estoi)





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
    # kldiv_mean, kldiv_interval = compute_mean(np_kldiv)
    delta_rmse_mean, delta_rmse_interval = compute_mean(np_delta_rmse)
    delta_rmse_interval = np.sqrt(np.var(np_delta_rmse))
    delta_sisdr_mean, delta_sisdr_interval = compute_mean(np_delta_sisdr)
    delta_sisdr_interval = np.sqrt(np.var(np_delta_sisdr))
    delta_pesq_mean, delta_pesq_interval = compute_mean(np_delta_pesq)
    delta_pesq_interval = np.sqrt(np.var(np_delta_pesq))
    delta_pesq_wb_mean, delta_pesq_wb_interval = compute_mean(np_delta_pesq_wb)
    delta_pesq_wb_interval = np.sqrt(np.var(np_delta_pesq_wb))
    delta_pesq_nb_mean, delta_pesq_nb_interval = compute_mean(np_delta_pesq_nb)
    delta_pesq_nb_interval = np.sqrt(np.var(np_delta_pesq_nb))
    delta_estoi_mean, delta_estoi_interval = compute_mean(np_delta_estoi)
    delta_estoi_interval = np.sqrt(np.var(np_delta_estoi))

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
        print('mean delta rmse score: {:.4f} +/- {:.4f}'.format(delta_rmse_mean, delta_rmse_interval), file=f)
        print('mean delta sisdr score: {:.1f} +/- {:.1f}'.format(delta_sisdr_mean, delta_sisdr_interval), file=f)
        print('mean delta pypesq score: {:.2f} +/- {:.2f}'.format(delta_pesq_mean, delta_pesq_interval), file=f)
        print('mean delta pesq wb score: {:.2f} +/- {:.2f}'.format(delta_pesq_wb_mean, delta_pesq_wb_interval), file=f)
        print('mean delta pesq nb score: {:.2f} +/- {:.2f}'.format(delta_pesq_nb_mean, delta_pesq_nb_interval), file=f)
        print('mean delta estoi score: {:.2f} +/- {:.2f}'.format(delta_estoi_mean, delta_estoi_interval), file=f)
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
    parser.add_argument('--phase', type=int, default=1, help='which phase to test')
    # parser.add_argument('--latent_check', action='store_true', help='whether to check latent space')
    parser.add_argument('--save_output', action='store_true', help='whether to save output files')
    args = parser.parse_args()

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    eval_metrics = EvalMetrics(metric=args.metric)

    models_path = args.state_dict_folder
    models_path = models_path.split('/')[-1]
    resfolder = args.resfolder + args.testset +  "_phase_" + str(args.phase) + '_' + args.outtype + '_' + models_path + '/'
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
    clean_encoder_path = [f for f in state_dict_paths if 'net_clean_encoder' in f]
    clean_encoder_path = args.state_dict_folder + '/' + clean_encoder_path[0]
    # clean_encoder_path = '/home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-03-11-10h15_complex_CVAE_causal=True_zdim=128_numsamples=5_kl_annflag=False_kl_epochs=20_klweight=0.00_miweight=0.00_skipc=False_skipuse=[5]_spadd=True_reconloss=multiple_recon=real_imag_reconweight=[1.0, 1.0, 1.0]/complex_CVAE_encoder_best_epoch.pt'
    # noise_encoder_path = [f for f in state_dict_paths if 'noise_encoder' in f]
    # noise_encoder_path = args.state_dict_folder + '/' + noise_encoder_path[0]


    clean_decoder_path = [f for f in state_dict_paths if 'net_clean_decoder' in f]
    clean_decoder_path = args.state_dict_folder + '/' + clean_decoder_path[0]
    # clean_decoder_path = '/home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-03-11-10h15_complex_CVAE_causal=True_zdim=128_numsamples=5_kl_annflag=False_kl_epochs=20_klweight=0.00_miweight=0.00_skipc=False_skipuse=[5]_spadd=True_reconloss=multiple_recon=real_imag_reconweight=[1.0, 1.0, 1.0]/complex_CVAE_decoder_best_epoch.pt'

    noise_decoder_path = [f for f in state_dict_paths if 'net_noise_decoder' in f]
    noise_decoder_path = args.state_dict_folder + '/' + noise_decoder_path[0]
    # noise_decoder_path = '/home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-03-14-13h17_complex_NVAE_causal=True_zdim=128_numsamples=5_kl_annflag=False_kl_epochs=20_klweight=0.00_miweight=0.00_skipc=False_skipuse=[0, 1, 2, 3, 4, 5]_spadd=False_reconloss=multiple_recon=real_imag_reconweight=[1.0, 1.0, 1.0]/complex_NVAE_decoder_best_epoch.pt'

    noisy_encoder_path = [f for f in state_dict_paths if 'noisy_encoder' in f]
    noisy_encoder_path = args.state_dict_folder + '/' + noisy_encoder_path[0]
    # noisy_encoder_path = '/home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-03-17-10h12_complex_NSVAE_causal=True_zdim=128_alpha=0.0000_wresi=0.00_wkl=1.0_numsamples=5_nsvae=original_latentnum=2_match=speech/complex_NSVAE_noisy_encoder_best_epoch.pt'

    noisy_clean_decoder_path = [f for f in state_dict_paths if 'noisy_clean_decoder' in f]
    noisy_clean_decoder_path = args.state_dict_folder + '/' + noisy_clean_decoder_path[0]
    # noisy_clean_decoder_path = clean_decoder_path
    try:
        noisy_noise_decoder_path = [f for f in state_dict_paths if 'noisy_noise_decoder' in f]
        noisy_noise_decoder_path = args.state_dict_folder + '/' + noisy_noise_decoder_path[0]
        # noisy_noise_decoder_path = noise_decoder_path
    except:
        pass

    # extract pretrain setups (skipc, recon_type, skipuse)
    pretrain_path = cfg.get("User","pre_clean_encoder")
    setups = pretrain_path.split('/')[-2]
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
    if 'causal' not in setups:
        causal = False
    if 'spadd' not in setups:
        spadd = False
    setups = setups.split('_')
    for s in setups:
        if 'zdim' in s:
            tmp = s.split('=')[-1]
            zdim = int(tmp)
        elif 'skipuse' in s:
            tmp = s.split('=')[-1][1:-1]
            tmp = tmp.split(', ')
            skipuse = []
            for n in tmp:
                skipuse.append(int(n))
        elif 'recon=' in s:
            pre_recon_type = s.split('=')[-1]
            if pre_recon_type == 'real':
                pre_recon_type = 'real_imag'
        elif 'causal=' in s:
            causal_tmp= s.split('=')[-1]
            causal = (causal_tmp.lower() == 'true')
        elif 'spadd=' in s:
            spadd_tmp = s.split('=')[-1]
            spadd = (spadd_tmp.lower() == 'true')
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()
    if spadd:
        clean_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
        clean_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=pre_recon_type, skip_to_use=skipuse)
        noise_model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
        noise_model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=pre_recon_type, skip_to_use=skipuse)
    else:
        clean_model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
        clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=pre_recon_type)
        noise_model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
        noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=pre_recon_type)
    # extract nsvae setups (w_resi, latentnum, nsvae_model, matching, zdim)
    nsvae_path = args.state_dict_folder.split('/')[-1]
    setups = nsvae_path.split('_')
    zdim = 0
    latent_num = 1
    if "skipc" not in setups:
        skipc = False
    if 'resyn=' not in setups:
        resyn = False
    for s in setups:
        if 'zdim' in s:
            tmp = s.split('=')[-1]
            zdim = int(tmp)
        elif 'latentnum' in s: # determine how many latent vector can be used
            tmp = s.split('=')[-1]
            latent_num = int(tmp)
        elif 'skipc' in s:
            skipc_tmp = s.split('=')[-1]
            skipc = (skipc_tmp.lower() == 'true')  
        elif 'resyn=' in s:
            resyn = s.split('=')[-1]
            resyn = (resyn.lower() == 'true')
        elif 'recontype=' in s:
            recon = s.split('=')[-1]
            if recon == 'real':
                recon = 'real_imag'
            else:
                recon = 'mask'          

    print("latent num ", latent_num)
    # load model
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()

    clean_model_encoder.load_state_dict(torch.load(clean_encoder_path, map_location=device))
    clean_model_decoder.load_state_dict(torch.load(clean_decoder_path, map_location=device))
    noise_model_decoder.load_state_dict(torch.load(noise_decoder_path, map_location=device))
    clean_model_encoder = clean_model_encoder.to(device)
    clean_model_encoder.eval()
    clean_model_decoder = clean_model_decoder.to(device)
    noise_model_decoder = noise_model_decoder.to(device)
    clean_model_decoder.eval()
    noise_model_decoder.eval()
    noisy_model_encoder = nsvae_pvae_dccrn_encoder_twophase(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples, 2)
    # noisy_model_encoder = nsvae_dccrn_encoder_original(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples, latent_num)
    noisy_model_encoder.load_state_dict(torch.load(noisy_encoder_path, map_location=device))
    noisy_model_encoder = noisy_model_encoder.to(device)
    noisy_model_encoder.eval()
    if latent_num == 1:
        noisy_clean_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon, use_sc=skipc, skip_to_use=skipuse, resynthesis=resyn) 
        # noisy_clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
        noisy_clean_model_decoder.load_state_dict(torch.load(noisy_clean_decoder_path, map_location=device))
        noisy_clean_model_decoder.to(device)
        noisy_clean_model_decoder.eval()

        noisy_noise_model_decoder = None
    elif latent_num == 2:
        noisy_clean_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, use_sc=skipc, skip_to_use=skipuse, resynthesis=resyn)
        # noisy_clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
        noisy_clean_model_decoder.load_state_dict(torch.load(noisy_clean_decoder_path, map_location=device))
        noisy_clean_model_decoder.to(device)
        noisy_noise_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, use_sc=skipc, skip_to_use=skipuse, resynthesis=resyn)
        noisy_clean_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon, use_sc=skipc, skip_to_use=skipuse, resynthesis=resyn)
        # noisy_clean_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
        noisy_clean_model_decoder.load_state_dict(torch.load(noisy_clean_decoder_path, map_location=device))
        noisy_clean_model_decoder.to(device)
        noisy_noise_model_decoder = nsvae_pvae_dccrn_decoder_twophase(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon, use_sc=skipc, skip_to_use=skipuse, resynthesis=resyn)
        # noisy_noise_model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
        noisy_noise_model_decoder.load_state_dict(torch.load(noisy_noise_decoder_path, map_location=device))
        noisy_noise_model_decoder.to(device)
        noisy_clean_model_decoder.eval()
        noisy_noise_model_decoder.eval()


    # load mean and std
    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')

    model_params = {
        'causal': causal,
        "skipc": skipc,
        "skipuse": skipuse,
        "recontype": recon,
        "latentnum": latent_num,
        "zdim": zdim,
        "num_samples": args.num_samples,
        "outtype": args.outtype,
        "latent_to_use": args.latent_to_use,
        "phase": args.phase
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

    run(clean_model_encoder, clean_model_decoder, noise_model_decoder, noisy_model_encoder, noisy_clean_model_decoder, noisy_noise_model_decoder,
        args.state_dict_folder, file_list, label_folder, args.testset, info_file,
        eval_metrics, STFT_dict, model_params, mean_file, std_file, resfolder, resjson, device, args.save_output)
