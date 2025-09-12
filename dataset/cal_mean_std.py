"""'

This is used calculate the mean and std of the whole data

Input: 1. The total connected noisy speech (note: it should be the noisy speech)
       2. The name of calculated mean
       3. The name of calculated std

output: 1. Calculated mean
        2. Calculated std

(Other training parameters can be changed in the code)
"""


from __future__ import print_function
import numpy as np
import librosa
from featurelib_r import calcFeat
import librosa
import os
import sys
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_folder_path)
from utils.read_config import myconf
from tqdm import tqdm

import argparse

# cfg = {'winlen': 512,
#        'nfft': 512,
#        'hopfrac': 256,
#        'fs': 16000,
#        'mingain': -80,
#        'feattype': 'LogPow'}  # LogPow PowSpec MagSpec


# Window_length = cfg['winlen']
# Fft_length = cfg['nfft']
# Frame_shift = cfg['hopfrac']
# fs_output = cfg['fs']


import os
import librosa
import numpy as np
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def process_file(file_path, cfg, Fft_length, Window_length, Frame_shift, fs_output):
    input_audio, fs_input = librosa.load(file_path, sr=None)

    if fs_input != fs_output:
        input_audio = librosa.resample(input_audio, fs_input, fs_output)

    stft_result = librosa.stft(input_audio, n_fft=Fft_length,
                               win_length=Window_length,
                               hop_length=Frame_shift,
                               window='hann')
    real = stft_result.real.T[..., None]
    imag = stft_result.imag.T[..., None]
    del stft_result
    del input_audio
    return np.concatenate((real, imag), axis=2)

def Cal_mean_std(cfg, folder_in, file_name_out_mean, file_name_out_std, n_jobs=4):
    Window_length = cfg.getint('STFT', 'winlen')
    Fft_length = cfg.getint('STFT', 'nfft')
    Frame_shift = cfg.getint('STFT', 'hopfrac')
    fs_output = cfg.getint('STFT', 'fs')

    file_list = librosa.util.find_files(folder_in, ext='wav')

    print("Processing files in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f, cfg, Fft_length, Window_length, Frame_shift, fs_output)
        for f in tqdm(file_list)
    )

    feat_array = np.concatenate(results, axis=0)

    del results

    sample_lps_mean = np.mean(feat_array, axis=0)
    sample_lps_std = np.std(feat_array, axis=0, ddof=1)

    np.savetxt(file_name_out_mean, sample_lps_mean)
    np.savetxt(file_name_out_std, sample_lps_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='cfg file')
    parser.add_argument('--folder', type=str, help='the folder of noisy speech')
    parser.add_argument('--file_name_out_mean', type=str, help='The name of calculated mean')
    parser.add_argument('--file_name_out_std', type=str, help='The name of calculated std')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')

    args = parser.parse_args()
    cfg = myconf()
    cfg.read(args.cfg_file)

    Cal_mean_std(cfg, args.folder, args.file_name_out_mean, args.file_name_out_std, n_jobs=args.n_jobs)


