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
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_folder_path)
from utils.read_config import myconf
from utils.eval_metrics import compute_median, EvalMetrics, compute_mean
from model.pvae_module import DCCRN_
import json
import model.net_config as net_config
import model.causal_netconfig as causal_netconfig

torch.manual_seed(0)
np.random.seed(0)



def run(model, statedict_folder, file_list, label_folder, testset, info_file, eval_metrics, STFT_dict, mean_file, std_file, 
        resfolder, resjson, save_output):
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
   
    
    idx = 0
    miu_real_array = []
    miu_imag_array = []
    for audio_file in tqdm(file_list):
        idx += 1
        if idx > 5:
            break
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
        # Reconstruction
        with torch.no_grad():
            estimated_clean, est_clean_stft = model(tmp_x, train=False)
            latent = model.std_DCCRN.latent
            miu = latent.view(latent.shape[0]* latent.shape[1], latent.shape[2], latent.shape[3])
            miu = miu.cpu().detach().numpy()
            miu_real_array.append(miu[...,0])
            miu_imag_array.append(miu[...,1])

        estimated_clean = estimated_clean.squeeze()
        estimated_clean = estimated_clean.cpu().detach().numpy()
        if eval_metrics.metric == 'no':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = 0, 0, 0, 0, 0, 0
        elif eval_metrics.metric == 'all':
            rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=estimated_clean, x_ref=clean_x, fs=fs_x, name=audio_file)
            rmse_noisy, sisdr_noisy, pesq_noisy, pesq_wb_noisy, pesq_nb_noisy, estoi_noisy = eval_metrics.eval(x_est=x, x_ref=clean_x, fs=fs_x, name=audio_file)
        else:
            metric_res = eval_metrics.eval(x_est=estimated_clean, x_ref=clean_x, fs=fs_x)
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
        # list_dis_mean.append(mean_dis.item())

        if save_output:
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + ".wav", x, fs_x)
            sf.write(data_check_dir + filename + "_pesq_" + str(pesq_wb) + "_enhanced.wav", estimated_clean, fs_x)

        
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
            'delta_estoi': estoi - estoi_noisy,
            # 'mean_dis': mean_dis.item()
        }
    miu_real_array = np.vstack(miu_real_array)
    miu_imag_array = np.vstack(miu_imag_array)

    numvecs, zdim = miu_real_array.shape[0], miu_real_array.shape[1]

    miu_real_mean = np.mean(miu_real_array, axis=0, keepdims=True)
    miu_imag_mean = np.mean(miu_imag_array, axis=0, keepdims=True)

    mean_dis = np.sqrt(np.sum(miu_real_mean**2+miu_imag_mean**2))

    center_miu_real = miu_real_array - miu_real_mean
    center_miu_imag = miu_imag_array - miu_imag_mean

    # cal cov(miu) -- vrr, vri, vii
    cov_miu_rr = np.matmul(center_miu_real.T, center_miu_real) / numvecs
    diag_elements = np.diagonal(cov_miu_rr, 0)
    l2_diag_perdim_rr = np.sum(np.abs(diag_elements)) / zdim
    offdiag = cov_miu_rr - np.diag(diag_elements)
    l2_offdiag_perdim_rr = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))
    cov_miu_ri = np.matmul(center_miu_real.T, center_miu_imag) / numvecs
    diag_elements = np.diagonal(cov_miu_ri, 0)
    l2_diag_perdim_ri = np.sum(np.abs(diag_elements)) / zdim
    offdiag = cov_miu_ri - np.diag(diag_elements)
    l2_offdiag_perdim_ri = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))
    cov_miu_ii = np.matmul(center_miu_imag.T, center_miu_imag) / numvecs
    diag_elements = np.diagonal(cov_miu_ii, 0)
    l2_diag_perdim_ii = np.sum(np.abs(diag_elements)) / zdim
    offdiag = cov_miu_ii - np.diag(diag_elements)
    l2_offdiag_perdim_ii = np.sum(np.abs(offdiag)) / (zdim * (zdim-1))

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

        print('mean delta rmse score: {:.4f} +/- {:.4f}'.format(delta_rmse_mean, delta_rmse_interval), file=f)
        print('mean delta sisdr score: {:.1f} +/- {:.1f}'.format(delta_sisdr_mean, delta_sisdr_interval), file=f)
        print('mean delta pypesq score: {:.2f} +/- {:.2f}'.format(delta_pesq_mean, delta_pesq_interval), file=f)
        print('mean delta pesq wb score: {:.2f} +/- {:.2f}'.format(delta_pesq_wb_mean, delta_pesq_wb_interval), file=f)
        print('mean delta pesq nb score: {:.2f} +/- {:.2f}'.format(delta_pesq_nb_mean, delta_pesq_nb_interval), file=f)
        print('mean delta estoi score: {:.2f} +/- {:.2f}'.format(delta_estoi_mean, delta_estoi_interval), file=f)

        print("cov miu", file=f)
        print('avg abs diag vrr per dim: {:.4f}'.format(l2_diag_perdim_rr), file=f)
        print('avg abs offdiag vrr per dim: {:.4f}'.format(l2_offdiag_perdim_rr), file=f)
        print('avg abs diag vri per dim: {:.4f}'.format(l2_diag_perdim_ri), file=f)
        print('avg abs offdiag vri per dim: {:.4f}'.format(l2_offdiag_perdim_ri), file=f)
        print('avg abs diag vii per dim: {:.4f}'.format(l2_diag_perdim_ii), file=f)
        print('avg abs offdiag vii per dim: {:.4f}'.format(l2_offdiag_perdim_ii), file=f)
        print('l2 dis of miu: {:.4f}'.format(mean_dis), file=f)

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
    parser.add_argument('--testset', type=str, default='dns2021', choices=['dns2021', 'wsj0', 'demand', 'dns2021_official','lowsnr_dns','lowsnr_wsj'], help='test on wsj or voicebank')
    parser.add_argument('--model_type', type=str, default='checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu')
    parser.add_argument('--resfolder', type=str, default='testres/', help='the folder to save results')
    parser.add_argument('--metric', type=str, default='all', help='the metric to calculate')
    parser.add_argument('--outtype', type=str, default='wiener', help='the method to obtain clean speech')
    parser.add_argument('--resjson', type=str, default='res.json', help='the json file to save evaluation results')
    parser.add_argument('--latent_check', action='store_true', help='whether to check latent space')
    parser.add_argument('--save_output', action='store_true', help='whether to save output files')
    args = parser.parse_args()

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    eval_metrics = EvalMetrics(metric=args.metric)


    # File path config
    if args.testset == 'dns2021_official':
        file_list = librosa.util.find_files('/data2/corpora/DNS1/datasets/test_set/synthetic/no_reverb/noisy', ext='wav')
        label_folder = '/data2/corpora/DNS1/datasets/test_set/synthetic/no_reverb/clean'
        info_file = None

    elif args.testset == 'dns2021':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/noisy_test_mix_short', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/dns/spk_split_dataset/clean_test_mix_short'
        info_file = None
    elif args.testset == 'lowsnr_dns':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/dns/spk_split_dataset/data_20h_lowsnr/test/noisy', ext='wav')
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
    elif args.testset == 'demand':
        file_list = librosa.util.find_files('/data1/corpora/jiatong_data/voicebank_demand/noisy_testset_wav_16k', ext='wav')
        label_folder = '/data1/corpora/jiatong_data/voicebank_demand/clean_testset_wav_16k'
        info_file = '/data1/corpora/jiatong_data/voicebank_demand/log_testset.txt'

    print(f'Test on {args.testset}, totl audio files {len(file_list)}')

    # load DVAE model
    state_folder = args.state_dict_folder
    cfg_file = os.path.join(state_folder, 'config.ini')
    cfg = myconf()
    cfg.read(cfg_file)

    # load model
    wlen = cfg.getint('STFT', 'winlen')
    hop = cfg.getint('STFT', 'hopfrac')
    fs = cfg.getint('STFT', 'fs')
    nfft = cfg.getint('STFT','nfft')
    setups = args.state_dict_folder.split('/')[-1]
    print(args.testset)
    resfolder = args.resfolder + args.testset + '_' + setups + '/'
    print(resfolder)
    res_json = resfolder + 'res.json'
    if 'skipuse' not in setups:
        skipuse = [0,1,2,3,4,5]
    if 'causal' not in setups:
        causal = False
    if 'datanorm' not in setups:
        datanorm = False
    if 'resynthesis' not in setups:
        resynthesis = False
    setups = setups.split('_')
    for s in setups:
        if 'skipuse' in s:
            tmp = s.split('=')[-1]
            skipuse = []
            for n in tmp:
                skipuse.append(int(n))
        elif 'causal=' in s:
            causal_tmp= s.split('=')[-1]
            causal = (causal_tmp.lower() == 'true')
        elif 'datanorm' in s:
            datanorm= s.split('=')[-1]
            datanorm = (datanorm.lower() == 'true')            
        elif 'recontype=' in s:
            tmp = s.split('=')[-1]
            if tmp != 'mask':
                recon_type='real_imag'
            else:
                recon_type = 'mask'
        elif 'resynthesis' in s:
            resynthesis= s.split('=')[-1]
            resynthesis = (resynthesis.lower() == 'true')  
    if not causal:
        net_params = net_config.get_net_params()  
    else:
        net_params = causal_netconfig.get_net_params()
    if datanorm:
        mean_file = "/home/jiatongl/dccrn-vae/dataset/mean_noisy_spksplit.txt"
        std_file = "/home/jiatongl/dccrn-vae/dataset/std_noisy_spksplit.txt"

        data_mean = np.loadtxt(mean_file)
        data_mean = data_mean.reshape(1, data_mean.shape[0], 1, data_mean.shape[1])
        data_mean = torch.tensor(data_mean, dtype=torch.float32).to(device)
        data_std = np.loadtxt(std_file)
        data_std = data_std.reshape(1, data_std.shape[0], 1, data_std.shape[1])
        data_std = torch.tensor(data_std, dtype=torch.float32).to(device)
    else:
        data_mean = None
        data_std = None        
    model = DCCRN_(nfft, hop, net_params,causal, device, wlen, skip_to_use=skipuse, recon_type=recon_type, resynthesis=resynthesis, data_mean=data_mean, data_std=data_std)

    # state dict paths
    if args.model_type == 'checkpoint':
        state_dict_paths = os.listdir(args.state_dict_folder)
        model_path = [f for f in state_dict_paths if 'checkpoint' in f]
        checkpoint = torch.load(args.state_dict_folder + '/' + model_path[0])
        curr_best = checkpoint['model_state_dict']
        model.load_state_dict(curr_best)
    elif args.model_type == 'final':
        state_dict_paths = os.listdir(args.state_dict_folder)
        model_path = [f for f in state_dict_paths if 'curr_best' in f]
        model.load_state_dict(torch.load(args.state_dict_folder + '/' + model_path[0],map_location=device))


    model = model.to(device)

    model.eval()

    print('encoder params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    # load mean and std
    mean_file = cfg.get('User','mean_file')
    std_file = cfg.get('User','std_file')


    # Load STFT parameters
    STFT_dict = {}
    STFT_dict['nfft'] = nfft
    STFT_dict['hop'] = hop
    STFT_dict['wlen'] = wlen

    print('='*80)
    print('STFT params')
    print(f'fs: {fs}')
    print(f'wlen: {wlen}')
    print(f'hop: {hop}')
    print(f'nfft: {nfft}')
    print('='*80)

    run(model, args.state_dict_folder, file_list, label_folder, args.testset, info_file,
        eval_metrics, STFT_dict, mean_file, std_file, resfolder, res_json, args.save_output)
