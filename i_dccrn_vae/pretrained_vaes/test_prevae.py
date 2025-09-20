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
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(main_folder_path)
from utils.read_config import myconf
from utils.eval_metrics import compute_median, EvalMetrics, compute_mean
from model.pvae_module import pvae_dccrn_encoder, pvae_dccrn_decoder, pvae_dccrn_decoder_no_skip, pvae_dccrn_encoder_no_skip, pvae_dccrn_encoder_skip_prepare, pvae_dccrn_decoder_skip_prepare
import json
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig

torch.manual_seed(0)
np.random.seed(0)

def cal_var_matrices(log_sigma, delta):

    sigma = torch.exp(log_sigma[..., 0])
    delta_real = delta[..., 0]
    delta_imag = delta[..., 1]

    # keep abs(delta) <= sigma (protection)
    abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + 1e-6)
    temp = sigma * 0.99 / (abs_delta + 1e-6)

    delta_real = torch.where(abs_delta >= (sigma-1e-3), delta_real * temp, delta_real)
    delta_imag = torch.where(abs_delta >= (sigma-1e-3), delta_imag * temp, delta_imag)

    vrr = 0.5 * (sigma + delta_real)
    vri = 0.5 * delta_imag
    vir = vri
    vii = 0.5 * (sigma - delta_real)

    return vrr, vri, vir, vii

def complex_kl(miu, log_sigma, delta):
    # input shape B,T,H,2

    miu_real = miu[:,:,:,0]
    miu_imag = miu[:,:,:,1]

    sigma = torch.exp(log_sigma[:,:,:,0])

    delta_real = delta[:,:,:,0]
    delta_imag = delta[:,:,:,1]

    miu_h_miu = torch.sum(miu_real.pow(2) + miu_imag.pow(2), dim=2)

    # keep abs(delta) <= sigma (protection)
    abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + 1e-6)
    temp = sigma * 0.99 / (abs_delta + 1e-6)

    delta_real = torch.where(abs_delta >= (sigma-1e-3), delta_real * temp, delta_real)
    delta_imag = torch.where(abs_delta >= (sigma-1e-3), delta_imag * temp, delta_imag)

    abs_delta_square = delta_real.pow(2) + delta_imag.pow(2)        

    log_sigma_2_minus_delta_2 = torch.log(sigma.pow(2) - abs_delta_square + 1e-6)

    res_kl = miu_h_miu + torch.sum(torch.abs(sigma - 1 - 0.5 * log_sigma_2_minus_delta_2), dim=2) # B, T

    # res_kl = torch.mean(torch.sum(res_kl, dim=1)) # per batch sample
    res_kl = torch.mean(res_kl) #per sample per time frame

    return res_kl


def run(model_encoder, model_decoder, testset, file_list, eval_metrics, STFT_dict, mean_file, std_file, 
        resfolder, resjson,device, save_outfiles, num_samples):
    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_pesq_wb = []
    list_pesq_nb = []
    list_estoi = []
    list_kldiv = []
    list_vrr = []
    list_vri = []
    list_vii = []
    list_zgivx_vrr_max = []
    list_zgivx_vrr_min = []
    list_zgivx_vri_max = []
    list_zgivx_vri_min = []
    list_zgivx_vii_max = []
    list_zgivx_vii_min = []
    list_zgivx_vrr_mean = []
    list_zgivx_vri_mean = []
    list_zgivx_vii_mean = []
    list_dis_mean = []
    # list_dis_var = []
    # list_recon = []

    data_check_dir = resfolder
    if not os.path.exists(data_check_dir):
        os.makedirs(data_check_dir)

    test_clean_speech_dict = {}
    # if latent_check:
    #     test_latent_dict = {}

    miu_real_array = []
    miu_imag_array = []

    fileidx = -1
    for audio_file in tqdm(file_list):

        fileidx += 1

        # root, file = os.path.split(audio_file)
        # filename, _ = os.path.splitext(file)
        if testset == 'dns' or testset == 'dnsoff':
            root, file = os.path.split(audio_file)
            filename, _ = os.path.splitext(file)
            file_id = audio_file.split('/')[-1]
            file_id = file_id.split('.')[0]
            p_id = file_id
            # p_id = file_id[:3]

        elif testset == 'wsj0':
            filefullname = audio_file.split('/')[-1]
            filefullname = filefullname.split('.')[0]
            clean_name = filefullname.split('_')[0]
            file_id = clean_name
            filename = filefullname
            p_id = filefullname.split('/')[-1]
        elif testset == 'demand':
            filefullname = audio_file.split('/')[-1]
            filefullname = filefullname.split('.')[0]
            file_id = filefullname
            filename = filefullname
            p_id = filefullname


        nfft = STFT_dict['nfft']
        hop = STFT_dict['hop']
        wlen = STFT_dict['wlen']
        # win = STFT_dict['win']
        trim = STFT_dict['trim']
        window = torch.hann_window(wlen)
        window = window.to(device)
        # feat_type = STFT_dict['feattype']

        x, fs_x = sf.read(audio_file)
        
        if trim:
            x, _ = librosa.effects.trim(x, top_db=30)
        
        #####################
        # preprocess the input
        #####################
        tmp_x = torch.from_numpy(x)
        tmp_x = tmp_x.float()
        tmp_x = tmp_x.to(device)
        tmp_x = tmp_x[None, ...]
        bs, time_len = tmp_x.shape

        # Reconstruction
        with torch.no_grad():
            z, miu, log_sigma, delta, skiper, C, F, stft_x = model_encoder(tmp_x, train=False)
            recon_sig, predict_stft = model_decoder(stft_x, z, skiper, C, F, train=False) # B * num_samples, time len

        vrr, vri, vir, vii = cal_var_matrices(log_sigma, delta) # b, t, zdim
        zgivx_vrr_min = torch.min(vrr)
        zgivx_vrr_max = torch.max(vrr)
        zgivx_vri_min = torch.min(vri)
        zgivx_vri_max = torch.max(vri)
        zgivx_vii_min = torch.min(vii)
        zgivx_vii_max = torch.max(vii)
        vrr = torch.mean(torch.mean(torch.abs(vrr), dim=0), dim=0)
        zgivx_vrr_mean = torch.mean(vrr)
        vrr = torch.sqrt(torch.sum(vrr.pow(2)))

        vri = torch.mean(torch.mean(torch.abs(vri), dim=0), dim=0)
        zgivx_vri_mean = torch.mean(vri)
        vri = torch.sqrt(torch.sum(vri.pow(2)))
        vii = torch.mean(torch.mean(torch.abs(vii), dim=0), dim=0)
        zgivx_vii_mean = torch.mean(vii)
        vii = torch.sqrt(torch.sum(vii.pow(2)))




        




        kl = complex_kl(miu, log_sigma, delta)
        recon_sig = torch.mean(recon_sig, dim=0)
        recon_res = recon_sig.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()



        miu = miu.view(miu.shape[0]* miu.shape[1], miu.shape[2], miu.shape[3])
        miu = miu.cpu().detach().numpy()
        miu_real_array.append(miu[...,0])
        miu_imag_array.append(miu[...,1])


        # resynthesis

        if eval_metrics.metric == 'no':
             rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = 0,0,0,0,0,0

        elif eval_metrics.metric == 'all':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=recon_res, x_ref=x, fs=fs_x, name=audio_file)
        else:
            metric_res = eval_metrics.eval(x_est=recon_res, x_ref=x, fs=fs_x, name=audio_file)
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
        list_kldiv.append(kl.item())
        list_vrr.append(vrr.item())
        list_vri.append(vri.item())
        list_vii.append(vii.item())

        list_zgivx_vrr_min.append(zgivx_vrr_min.item())
        list_zgivx_vrr_max.append(zgivx_vrr_max.item())
        list_zgivx_vri_min.append(zgivx_vri_min.item())
        list_zgivx_vri_max.append(zgivx_vri_max.item())
        list_zgivx_vii_min.append(zgivx_vii_min.item())
        list_zgivx_vii_max.append(zgivx_vii_max.item())

        list_zgivx_vrr_mean.append(zgivx_vrr_mean.item())
        list_zgivx_vri_mean.append(zgivx_vri_mean.item())
        list_zgivx_vii_mean.append(zgivx_vii_mean.item())


        if save_outfiles:
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + ".wav", x, fs_x)
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_recon.wav", recon_res, fs_x)
        
        test_clean_speech_dict[file_id] = {
            'p_id': p_id,
            'utt_name': file_id,
            'noise_type': 'clean',
            'snr': 100,
            'sisdr': sisdr,
            'pesq': pesq,
            'pesq_wb': pesq_wb,
            'pesq_nb': pesq_nb,
            'estoi': estoi,
            'kl_div': kl.item(),
            'vrr': vrr.item(),
            'vri': vri.item(),
            'vii': vii.item(),
            'zgivx_vrr_min': zgivx_vrr_min.item(),
            'zgivx_vrr_max': zgivx_vrr_max.item(),
            'zgivx_vri_min': zgivx_vri_min.item(),
            'zgivx_vri_max': zgivx_vri_max.item(),
            'zgivx_vii_min': zgivx_vii_min.item(),
            'zgivx_vii_max': zgivx_vii_max.item(),
            'zgivx_vrr_mean': zgivx_vrr_mean.item(),
            'zgivx_vri_mean': zgivx_vri_mean.item(),
            'zgivx_vii_mean': zgivx_vii_mean.item()

        }



    miu_real_array = np.vstack(miu_real_array)
    miu_imag_array = np.vstack(miu_imag_array)

    numvecs, zdim = miu_real_array.shape[0], miu_real_array.shape[1]

    miu_real_mean = np.mean(miu_real_array, axis=0, keepdims=True)
    miu_imag_mean = np.mean(miu_imag_array, axis=0, keepdims=True)

    mean_dis = np.sqrt(np.sum(miu_real_mean**2+miu_imag_mean**2))

    center_miu_real = miu_real_array - miu_real_mean
    center_miu_imag = miu_imag_array - miu_imag_mean

    # Plot only diagonal elements
    plt.figure(figsize=(8, 5))
    # cal cov(miu) -- vrr, vri, vii
    cov_miu_rr = np.matmul(center_miu_real.T, center_miu_real) / numvecs
    diag_elements = np.diagonal(cov_miu_rr, 0)
    plt.plot(diag_elements, label="RR Diag", marker='o')
    # print('rr diag',np.max(np.abs(diag_elements)), np.min(np.abs(diag_elements)))
    l2_diag_perdim_rr = np.sum(np.abs(diag_elements)) / zdim
    diag_rr_max = np.max(diag_elements)
    diag_rr_min = np.min(diag_elements)
    diag_rr_mean = np.mean(diag_elements)
    offdiag = cov_miu_rr - np.diag(diag_elements)
    # print('rr offdiag',np.max(np.abs(offdiag)), np.min(np.abs(offdiag)))
    l2_offdiag_perdim_rr = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))
    cov_miu_ri = np.matmul(center_miu_real.T, center_miu_imag) / numvecs
    diag_elements = np.diagonal(cov_miu_ri, 0)
    plt.plot(diag_elements, label="RI Diag", marker='s')
    # print('ri diag',np.max(np.abs(diag_elements)), np.min(np.abs(diag_elements)))
    l2_diag_perdim_ri = np.sum(np.abs(diag_elements)) / zdim
    diag_ri_max = np.max(diag_elements)
    diag_ri_min = np.min(diag_elements)
    diag_ri_mean = np.mean(diag_elements)
    offdiag = cov_miu_ri - np.diag(diag_elements)
    # print('ri offdiag',np.max(np.abs(offdiag)), np.min(np.abs(offdiag)))
    l2_offdiag_perdim_ri = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))
    cov_miu_ii = np.matmul(center_miu_imag.T, center_miu_imag) / numvecs
    diag_elements = np.diagonal(cov_miu_ii, 0)
    plt.plot(diag_elements, label="II Diag", marker='^')
    # print('ii diag',np.max(np.abs(diag_elements)), np.min(np.abs(diag_elements)))
    l2_diag_perdim_ii = np.sum(np.abs(diag_elements)) / zdim
    diag_ii_max = np.max(diag_elements)
    diag_ii_min = np.min(diag_elements)
    diag_ii_mean = np.mean(diag_elements)
    offdiag = cov_miu_ii - np.diag(diag_elements)
    # print('ii offdiag',np.max(np.abs(offdiag)), np.min(np.abs(offdiag)))
    l2_offdiag_perdim_ii = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))

    # Labels and title
    plt.title("Diagonal Elements of Covariance Matrices")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim((-0.5,0.5))
    plt.legend()
    plt.grid()

    # Save the figure instead of displaying it
    plt.savefig(resfolder + "diagonal_elements_plot.png", dpi=300, bbox_inches='tight')




    np_rmse = np.array(list_rmse)
    np_sisdr = np.array(list_sisdr)
    np_pesq = np.array(list_pesq)
    np_pesq_wb = np.array(list_pesq_wb)
    np_pesq_nb = np.array(list_pesq_nb)
    np_estoi = np.array(list_estoi)
    np_kldiv = np.array(list_kldiv)
    np_vrr = np.array(list_vrr)
    np_vri = np.array(list_vri)
    np_vii = np.array(list_vii)
    np_zgivx_vrr_min = np.array(list_zgivx_vrr_min)
    np_zgivx_vrr_max = np.array(list_zgivx_vrr_max)
    np_zgivx_vri_min = np.array(list_zgivx_vri_min)
    np_zgivx_vri_max = np.array(list_zgivx_vri_max)
    np_zgivx_vii_min = np.array(list_zgivx_vii_min)
    np_zgivx_vii_max = np.array(list_zgivx_vii_max)
    np_zgivx_vrr_mean = np.array(list_zgivx_vrr_mean)
    np_zgivx_vri_mean = np.array(list_zgivx_vri_mean)
    np_zgivx_vii_mean = np.array(list_zgivx_vii_mean)


    file_path = resjson

    # Write the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(test_clean_speech_dict, json_file, indent=4)
    

    rmse_mean, rmse_interval = compute_mean(np_rmse)
    sisdr_mean, sisdr_interval = compute_mean(np_sisdr)
    pesq_mean, pesq_interval = compute_mean(np_pesq)
    pesq_wb_mean, pesq_wb_interval = compute_mean(np_pesq_wb)
    pesq_nb_mean, pesq_nb_interval = compute_mean(np_pesq_nb)
    estoi_mean, estoi_interval = compute_mean(np_estoi)
    kldiv_mean, kldiv_interval = compute_mean(np_kldiv)
    vrr_mean, vrr_interval = compute_mean(np_vrr)
    vrr_interval = np.sqrt(np.var(np_vrr))
    vri_mean, vri_interval = compute_mean(np_vri)
    vri_interval = np.sqrt(np.var(np_vri))
    vii_mean, vii_interval = compute_mean(np_vii)
    vii_interval = np.sqrt(np.var(np_vii))

    zgivx_vrr_min_mean, _ = compute_mean(np_zgivx_vrr_min)
    zgivx_vrr_max_mean, _ = compute_mean(np_zgivx_vrr_max)
    zgivx_vri_min_mean, _ = compute_mean(np_zgivx_vri_min)
    zgivx_vri_max_mean, _ = compute_mean(np_zgivx_vri_max)
    zgivx_vii_min_mean, _ = compute_mean(np_zgivx_vii_min)
    zgivx_vii_max_mean, _ = compute_mean(np_zgivx_vii_max)
    zgivx_vrr_mean_mean, _ = compute_mean(np_zgivx_vrr_mean)
    zgivx_vri_mean_mean, _ = compute_mean(np_zgivx_vri_mean)
    zgivx_vii_mean_mean, _ = compute_mean(np_zgivx_vii_mean)

    zgivx_vrr_min_interval = np.sqrt(np.var(np_zgivx_vrr_min))
    zgivx_vrr_max_interval = np.sqrt(np.var(np_zgivx_vrr_max))
    zgivx_vri_min_interval = np.sqrt(np.var(np_zgivx_vri_min))
    zgivx_vri_max_interval = np.sqrt(np.var(np_zgivx_vri_max))
    zgivx_vii_min_interval = np.sqrt(np.var(np_zgivx_vii_min))
    zgivx_vii_max_interval = np.sqrt(np.var(np_zgivx_vii_max))
    zgivx_vrr_mean_interval = np.sqrt(np.var(np_zgivx_vrr_mean))
    zgivx_vri_mean_interval = np.sqrt(np.var(np_zgivx_vri_mean))
    zgivx_vii_mean_interval = np.sqrt(np.var(np_zgivx_vii_mean))

    with open(data_check_dir+'log.txt', 'w') as f:
        print('Re-synthesis finished', file=f) 
        print("mean evaluation", file=f)
        print('mean rmse score: {:.4f} +/- {:.4f}'.format(rmse_mean, rmse_interval), file=f)
        print('mean sisdr score: {:.1f} +/- {:.1f}'.format(sisdr_mean, sisdr_interval), file=f)
        print('mean pypesq score: {:.2f} +/- {:.2f}'.format(pesq_mean, pesq_interval), file=f)
        print('mean pesq wb score: {:.2f} +/- {:.2f}'.format(pesq_wb_mean, pesq_wb_interval), file=f)
        print('mean pesq nb score: {:.2f} +/- {:.2f}'.format(pesq_nb_mean, pesq_nb_interval), file=f)
        print('mean estoi score: {:.2f} +/- {:.2f}'.format(estoi_mean, estoi_interval), file=f)
        print('mean kldiv score: {:.4f} +/- {:.4f}'.format(kldiv_mean, kldiv_interval), file=f)
        print('mean vrr score: {:.4f} +/- {:.4f}'.format(vrr_mean, vrr_interval), file=f)
        print('mean vri score: {:.4f} +/- {:.4f}'.format(vri_mean, vri_interval), file=f)
        print('mean vii score: {:.4f} +/- {:.4f}'.format(vii_mean, vii_interval), file=f)

        print('zgivx_vrr_min: {:.4f} +/- {:.4f}'.format(zgivx_vrr_min_mean, zgivx_vrr_min_interval), file=f)
        print('zgivx_vrr_max: {:.4f} +/- {:.4f}'.format(zgivx_vrr_max_mean, zgivx_vrr_max_interval), file=f)
        print('zgivx_vri_min: {:.4f} +/- {:.4f}'.format(zgivx_vri_min_mean, zgivx_vri_min_interval), file=f)
        print('zgivx_vri_max: {:.4f} +/- {:.4f}'.format(zgivx_vri_max_mean, zgivx_vri_max_interval), file=f)
        print('zgivx_vii_min: {:.4f} +/- {:.4f}'.format(zgivx_vii_min_mean, zgivx_vii_min_interval), file=f)
        print('zgivx_vii_max: {:.4f} +/- {:.4f}'.format(zgivx_vii_max_mean, zgivx_vii_max_interval), file=f)
        print('zgivx_vrr_mean: {:.4f} +/- {:.4f}'.format(zgivx_vrr_mean_mean, zgivx_vrr_mean_interval), file=f)
        print('zgivx_vri_mean: {:.4f} +/- {:.4f}'.format(zgivx_vri_mean_mean, zgivx_vri_mean_interval), file=f)
        print('zgivx_vii_mean: {:.4f} +/- {:.4f}'.format(zgivx_vii_mean_mean, zgivx_vii_mean_interval), file=f)
        # print('mean miu dis score: {:.4f} +/- {:.4f}'.format(miu_mean, miu_interval), file=f)

        print("cov miu", file=f)
        print('avg abs diag vrr per dim: {:.4f}'.format(l2_diag_perdim_rr), file=f)
        print('diag min vrr per dim: {:.4f}'.format(diag_rr_min), file=f)
        print('diag max vrr per dim: {:.4f}'.format(diag_rr_max), file=f)
        print('diag mean (noabs) vrr per dim: {:.4f}'.format(diag_rr_mean), file=f)
        print('avg abs offdiag vrr per dim: {:.4f}'.format(l2_offdiag_perdim_rr), file=f)
        print('avg abs diag vri per dim: {:.4f}'.format(l2_diag_perdim_ri), file=f)
        print('diag min vri per dim: {:.4f}'.format(diag_ri_min), file=f)
        print('diag max vri per dim: {:.4f}'.format(diag_ri_max), file=f)
        print('diag mean (noabs) vri per dim: {:.4f}'.format(diag_ri_mean), file=f)
        print('avg abs offdiag vri per dim: {:.4f}'.format(l2_offdiag_perdim_ri), file=f)
        print('avg abs diag vii per dim: {:.4f}'.format(l2_diag_perdim_ii), file=f)
        print('diag min vii per dim: {:.4f}'.format(diag_ii_min), file=f)
        print('diag max vii per dim: {:.4f}'.format(diag_ii_max), file=f)
        print('diag mean (noabs) vii per dim: {:.4f}'.format(diag_ii_mean), file=f)
        print('avg abs offdiag vii per dim: {:.4f}'.format(l2_offdiag_perdim_ii), file=f)
        print('l2 dis of miu: {:.4f}'.format(mean_dis), file=f)
        
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict_folder', type=str, default=None, help='pretrained model state')
    parser.add_argument('--testset', type=str, default='dns2021', choices=['dns', 'wsj0', 'demand', 'dnsoff'], help='test on wsj or voicebank')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu')
    parser.add_argument('--resfolder', type=str, default='testres/', help='the folder to save results')
    parser.add_argument('--metric', type=str, default='all', help='the metric to calculate')
    parser.add_argument('--num_samples', type=int, default=1, help='how many z sampled in evaluation')
    parser.add_argument('--resjson', type=str, default='res.json', help='the json file to save evaluation results')
    # parser.add_argument('--latent_check', action='store_true', help='whether to save latent vectors')
    parser.add_argument('--save_outfiles', action='store_true', help='whether to save output files')
    args = parser.parse_args()

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    eval_metrics = EvalMetrics(metric=args.metric)

    models_path = args.state_dict_folder.split('/')[-1]
    resfolder = args.resfolder + args.testset + '_' + models_path + '/'
    resjson = resfolder + 'res.json'
    print(args.state_dict_folder)
    # File path config
    if args.testset == 'dns':
        if 'CVAE' in args.state_dict_folder:
            file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/clean_test_mix_short', ext='wav')
        elif 'NVAE' in args.state_dict_folder:
            file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/noise_test_mix_short', ext='wav')
    elif args.testset == 'dnsoff':
        file_list = librosa.util.find_files('/fs/dss/work/geck5401/official_test/dns2021_official_noreverb/clean', ext='wav')
        # file_list = librosa.util.find_files('/data2/jiatong_data/WSJ0/val_debug', ext='wav')
    elif args.testset == 'wsj0':
        file_list = librosa.util.find_files('/fs/dss/work/geck5401/official_test/wsj0_qut/test_si_et_05', ext='wav')
    elif args.testset == 'demand':
        file_list = librosa.util.find_files('/fs/dss/work/geck5401/official_test/vbdmd/clean_testset_wav_16k', ext='wav')

    print(f'Test on {args.testset}, totl audio files {len(file_list)}')

    # load DVAE model
    # state_file = args.state_dict_encoder
    model_folder = args.state_dict_folder
    cfg_file = os.path.join(model_folder, 'config.ini')
    cfg = myconf()
    cfg.read(cfg_file)

    # load model
    # get zdim
    model_cfg = model_folder.split('_')
    zdim_cfg = [c for c in model_cfg if 'zdim' in c]
    zdim_cfg = zdim_cfg[0].split('=')
    zdim = int(zdim_cfg[1])
    z_dim = zdim

    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')

    trim = cfg.getboolean('STFT', 'trim')

    setups = model_folder
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
    if 'causal' not in setups:
        causal = False

    if 'spad' not in setups:
        spad = False
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
        elif 'spad' in s:
            spad_tmp = s.split('=')[-1]
            spad = (spad_tmp.lower() == 'true')
        
    if not causal:
        net_params = net_config.get_net_params()
    else:
        net_params = causal_netconfig.get_net_params()
    if skipc == 'False':
        if not spad:
            model_encoder = pvae_dccrn_encoder_no_skip(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
            model_decoder = pvae_dccrn_decoder_no_skip(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type)
        else:
            model_encoder = pvae_dccrn_encoder_skip_prepare(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
            model_decoder = pvae_dccrn_decoder_skip_prepare(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse)            
    else:
        model_encoder = pvae_dccrn_encoder(net_params, causal, device, zdim, nfft, hop, wlen, args.num_samples)
        model_decoder = pvae_dccrn_decoder(net_params, causal, device, args.num_samples, zdim, nfft, hop, wlen, recon_type=recon_type, skip_to_use=skipuse)

    state_dict_paths = os.listdir(model_folder)
    encoder_path = [f for f in state_dict_paths if 'encoder_best_epoch' in f]
    encoder_path = model_folder + '/' + encoder_path[0]
    decoder_path = [f for f in state_dict_paths if 'decoder_best_epoch' in f]
    decoder_path = args.state_dict_folder + '/' + decoder_path[0]
    model_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    model_encoder.to(device)
    model_decoder.to(device)
    model_encoder.eval()
    model_decoder.eval()
    print('encoder params: %.2fM' % (sum(p.numel() for p in model_encoder.parameters()) / 1e6))
    print('decoder params: %.2fM' % (sum(p.numel() for p in model_decoder.parameters()) / 1e6))

    # load mean and std
    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')


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

    run(model_encoder, model_decoder, args.testset, file_list, eval_metrics, STFT_dict, mean_file, std_file, 
        resfolder, resjson, device, args.save_outfiles, args.num_samples)
