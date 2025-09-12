# coding: utf-8
# Author：WangTianRui
# Date ：2020/9/30 10:55


from model.complex_progress import *
# import torchaudio_contrib as audio_nn
# from utils import *
# import matplotlib.pyplot as plt


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, device):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)
        self.window = self.window.to(device)
        # self.stft = torch.stft(n_fft=self.n_fft, hop_length=self.hop_length, win_length=win_length, window=torch.hann_window(win_length),return_complex=False)

    def forward(self, signal):
        x = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window,return_complex=True) # batch, freq, time, 2 (real and imag)
        x = torch.view_as_real(x)
            # mag, phase = audio_nn.magphase(x, power=1.)

        # mix = torch.stack((mag, phase), dim=-1)
        return x


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, device):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.window = torch.hann_window(self.win_length)
        self.window = self.window.to(device)
        # self.istft = torch.istft(n_fft=self.n_fft, hop_length=self.hop_length, win_length=win_length, window=torch.hann_window(win_length),return_complex=False)

    def forward(self, x):
        # B, C, F, T, D = x.shape
        # x = x.view(B, F, T, D)
        x_istft = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window,return_complex=False)
        return x_istft


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None, causal=False):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        if not causal:
            self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        else:
            self.conv = causal_complex_conv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                        stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()
    def check_and_log_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
            # self.detect_anormal = False
    def forward(self, x, train):
        x = self.conv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x
 
  

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None, causal=False, if_bn=True):
        super().__init__()
        if not causal:
            self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                    stride=stride, padding=padding)
        else:
            self.transconv = causal_ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                    stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()
        self.if_bn = if_bn
    def check_and_log_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
    def forward(self, x, train=True):
        x = self.transconv(x)
        if self.if_bn:
            x = self.bn(x, train)
            x = self.prelu(x)
        return x


class standard_DCCRN(nn.Module):
    def __init__(self, net_params, causal, device, skip_to_use):
        super().__init__()
        self.device = device
        self.causal = causal
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params["dense"][0], net_params["dense"][1])
        self.skip_to_use = skip_to_use
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            # self.add_module("encoder{%d}" % index, model)
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_dims[index + 1],
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)
            # self.add_module("lstm{%d}" % index, model)
        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            if index in self.skip_to_use:
                in_decoder_channel = de_channels[index] + en_channels[len(self.encoders) - index]
            else:
                in_decoder_channel = de_channels[index]
            if index == len(de_channels) - 2:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal, if_bn=True
                )
            else:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal
                )
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        self.decoders = nn.ModuleList(self.decoders)
        self.linear = ComplexConv2d(in_channel=1, out_channel=1, kernel_size=1, stride=1)

        self.detect_anormal = True

    def check_and_log_nan(self, tensor, name, input1=None, input2= None):
        if torch.isnan(tensor).any() and self.detect_anormal:
            print(f"NaN detected in {name}")
            if input1 != None:
                nan_indices = torch.isnan(tensor).nonzero(as_tuple=False)
                print("input1 nan track")
                print(input1[nan_indices])
                print("input2 nan track")
                print(input2[nan_indices])
            raise RuntimeError(f"NaN detected in {name}")
            # self.detect_anormal = False

    def forward(self, x, train=True):
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) #B, CF; T;D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T;B;CF;D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T;B;H;D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B;T;H;D
        if not train:
            self.latent = lstm_
        lstm_out = lstm_.reshape(B * T, -1, D) #BT;H;D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B;C;F;T;D
        for index, decoder in enumerate(self.decoders):
            if index in self.skip_to_use:
                p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
            p = decoder(p, train)

        return p

class DCCRN_(nn.Module):
    def __init__(self, n_fft, hop_len, net_params, causal, device, win_length, skip_to_use, recon_type, resynthesis, data_mean, data_std):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        self.std_DCCRN = standard_DCCRN(net_params, causal, device=device, skip_to_use=skip_to_use)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)
        self.recon_type = recon_type
        self.resynthesis = resynthesis
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False

    def forward(self, signal, train=True):
        stft = self.stft(signal) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        if self.datanorm:
            stft = (stft - self.data_mean) / (self.data_std + 1e-6)
            stft = stft.clone()
            stft[:,0,:,1] = 0
            stft[:,-1,:,1] = 0
        stft = torch.unsqueeze(stft, 1)
        out = self.std_DCCRN(stft, train=train)
        if self.recon_type == 'mask':
            mask = out
            mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
            mask_mag = torch.tanh(mask_mag)
            real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
            imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            input_mag = torch.sqrt(stft[:,:,:,:,0].pow(2)+stft[:,:,:,:,1].pow(2))
            input_phase = torch.arctan2(stft[:,:,:,:,1], stft[:,:,:,:,0])
            predict = input_mag * mask_mag * torch.exp(1j * (input_phase + mask_phase))
            predict = torch.squeeze(predict, 1)
            if self.datanorm:
                predict = torch.view_as_real(predict) # b,f,t,2
                predict = self.data_std * predict + self.data_mean
                predict = torch.complex(predict[...,0],predict[...,1])
            clean = self.istft(predict)
            if self.resynthesis:
                resyn_stft = self.stft(clean)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])

            # real and imag
        if self.recon_type == 'real_imag':
            out = torch.squeeze(out, 1)# B*numsamples, freq, time, 2
            if self.datanorm:
                out = self.data_std * out + self.data_mean
            predict = torch.complex(out[:,:,:,0],out[:,:,:,1]) #B*numsamples, freq, time
            clean = self.istft(predict) # B * numsamples, time_len         
            if self.resynthesis:
                resyn_stft = self.stft(clean)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])   

        return clean, predict
       


class pvae_dccrn_encoder(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, data_mean=None, data_std=None):
        super().__init__()
        self.device = device
        self.causal = causal
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = int(3 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False


        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        if self.datanorm:
            stft_x = (stft_x - self.data_mean) / (self.data_std + 1e-6)
            stft_x = stft_x.clone()
            stft_x[:,0,:,1] = 0
            stft_x[:,-1,:,1] = 0
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D
        
        miu = lstm_[:, :, 0:self.zdim, :]
        log_sigma = lstm_[:, :, self.zdim:2*self.zdim, :]
        delta = lstm_[:, :, 2*self.zdim:, :]

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2

        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x

class pvae_dccrn_decoder(nn.Module):
    def __init__(self, net_params, causal, device, num_samples, zdim, n_fft, hop_len, win_length, recon_type, skip_to_use, resynthesis=False, data_mean=None,data_std=None):
        super().__init__()
        self.device = device
        self.causal = causal
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.num_samples = num_samples
        self.zdim = zdim
        self.recon_type = recon_type
        self.skip_to_use = skip_to_use
        self.resynthesis = resynthesis
        self.dense = ComplexDense(zdim, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        decoder_chw = net_params["decoder_chw"]

        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            if index in skip_to_use:
                in_decoder_channel = de_channels[index] + en_channels[len(en_channels) - 1 - index]
            else:
                in_decoder_channel = de_channels[index]
            if index == len(de_channels) - 2:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal, if_bn=True
                )
            else:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal
                )                
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.decoders = nn.ModuleList(self.decoders)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)
        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False

    def forward(self, stft_x, z, skiper, C, F, train=True):

        B_numsamples, T, zdim, D = z.shape
        lstm_out = z.reshape(B_numsamples * T, -1, D) # B * self.num_samples * T, zdim, D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B_numsamples, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B * self.num_samples, C, F, T, D
        for index, decoder in enumerate(self.decoders):
            # if index % 2 != 0:
            if index in self.skip_to_use:
                skiper_concate = skiper[len(skiper) - index - 1]
                tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
                skiper_concate = skiper_concate.unsqueeze(1)
                skiper_concate = skiper_concate.repeat(1, self.num_samples, 1, 1, 1, 1)
                skiper_concate = skiper_concate.view(B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d)
                p = torch.cat([p, skiper_concate], dim=1)
            p = decoder(p, train)


        recon_stft = p # B * self.num_samples, 1, F, T, D=2
        # mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        # mask_mag = torch.tanh(mask_mag)
        # real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        # imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        # mask_phase = torch.atan2(imag_phase, real_phase)

        # either recon magnitude and phase or recon real and imag
        # mag and phase
        # recon_mag = torch.sqrt(recon_stft[:,:,:,:,0].pow(2)+recon_stft[:,:,:,:,1].pow(2)) # B*numsamples, 1, freq, time, D
        # recon_phase = torch.arctan2(recon_stft[:,:,:,:,1], recon_stft[:,:,:,:,0]) # may be nan TODO: add protection
        # predict = recon_mag * torch.exp(1j * (recon_phase))        
        # predict = torch.squeeze(predict, 1)
        # recon_sig = self.istft(predict) # B * numsamples, time_len

        # real and imag
        if self.recon_type == 'real_imag':
            predict = torch.squeeze(recon_stft, 1)# B*numsamples, freq, time
            if self.datanorm:
                predict = self.data_std * predict + self.data_mean
            predict = torch.complex(predict[:,:,:,0],predict[:,:,:,1]) #B*numsamples, 1, freq, time
            recon_sig = self.istft(predict) # B * numsamples, time_len
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])

        if self.recon_type == 'mask':
            mask = recon_stft
            mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
            mask_mag = torch.tanh(mask_mag)
            real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
            imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            b,freq, time, dim = stft_x.shape
            stft_x = stft_x.unsqueeze(1)
            stft_x = stft_x.repeat(1,self.num_samples, 1, 1, 1)
            stft_x = stft_x.view(b * self.num_samples, freq, time, dim)
            stft_x = stft_x.unsqueeze(1)
            input_mag = torch.sqrt(stft_x[:,:,:,:,0].pow(2)+stft_x[:,:,:,:,1].pow(2))
            input_phase = torch.arctan2(stft_x[:,:,:,:,1], stft_x[:,:,:,:,0])
            predict = input_mag * mask_mag * torch.exp(1j * (input_phase + mask_phase))         
            predict = torch.squeeze(predict, 1)
            if self.datanorm:
                predict = torch.view_as_real(predict) # b,f,t,2
                predict = self.data_std * predict + self.data_mean
                predict = torch.complex(predict[...,0],predict[...,1])
            recon_sig = self.istft(predict)
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])
        return recon_sig, predict





class pvae_dccrn_encoder_no_skip(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, data_mean, data_std):
        super().__init__()
        self.device = device
        self.causal = causal
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = int(3 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False


        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps

        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        if self.datanorm:
            stft_x = (stft_x - self.data_mean) / (self.data_std + 1e-6)
            stft_x = stft_x.clone()
            stft_x[:,0,:,1] = 0
            stft_x[:,-1,:,1] = 0
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D
        
        miu = lstm_[:, :, 0:self.zdim, :]
        log_sigma = lstm_[:, :, self.zdim:2*self.zdim, :]
        delta = lstm_[:, :, 2*self.zdim:, :]

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x
    

class pvae_dccrn_encoder_no_skip_fc_latent(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, data_mean, data_std):
        super().__init__()
        self.device = device
        self.causal = causal
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense_mean = ComplexDense(dense_inputsize, zdim)
        self.dense_logvar = ComplexDense(dense_inputsize, zdim)
        self.dense_delta = ComplexDense(dense_inputsize, zdim)
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = self.zdim
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False


        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(torch.clamp(log_sigma[:,:,:,0], -13, 13))
        # real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)

        # all of denominators and nominators are of size B, T, H
        sqrt_arg = torch.clamp(2 * (real_sigma + real_delta), min=self.epsilon)
        denominator = torch.sqrt(sqrt_arg)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        sqrt_arg = torch.clamp(real_sigma.pow(2)-abs_delta.pow(2), min=self.epsilon)
        imag_scale_y = torch.sqrt(sqrt_arg) / (denominator)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        if self.datanorm:
            stft_x = (stft_x - self.data_mean) / (self.data_std + 1e-6)
            stft_x = stft_x.clone()
            stft_x[:,0,:,1] = 0
            stft_x[:,-1,:,1] = 0
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D

        miu = self.dense_mean(lstm_)
        log_sigma = self.dense_logvar(lstm_)
        delta = self.dense_delta(lstm_)

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x



class pvae_dccrn_decoder_no_skip(nn.Module):
    def __init__(self, net_params, causal, device, num_samples, zdim, n_fft, hop_len, win_length, recon_type, resynthesis=False, data_mean=None, data_std=None):
        super().__init__()
        self.device = device
        self.causal = causal
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.num_samples = num_samples
        self.zdim = zdim
        self.recon_type = recon_type
        self.resynthesis = resynthesis
        self.dense = ComplexDense(zdim, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        decoder_chw = net_params["decoder_chw"]

        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            # if index % 2 != 0:
            #     in_decoder_channel = de_channels[index] + en_channels[len(en_channels) - 1 - index]
            # else:
            in_decoder_channel = de_channels[index]
            if index != len(de_channels) - 2:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal
                )
            else:
                model = Decoder(
                    in_channel= in_decoder_channel,
                    out_channel=de_channels[index + 1],
                    kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                    chw=decoder_chw[index], causal=self.causal, if_bn=True
                )
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.decoders = nn.ModuleList(self.decoders)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        if self.data_mean is not None and self.data_std is not None:
            self.datanorm = True
        else:
            self.datanorm = False

    def forward(self, stft_x, z, skiper, C, F, train=True):

        B_numsamples, T, zdim, D = z.shape
        lstm_out = z.reshape(B_numsamples * T, -1, D) # B * self.num_samples * T, zdim, D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B_numsamples, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B * self.num_samples, C, F, T, D
        # self.decoderout = []
        for index, decoder in enumerate(self.decoders):
            # if index % 2 != 0:
            #     skiper_concate = skiper[len(skiper) - index - 1]
            #     tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
            #     skiper_concate = skiper_concate.unsqueeze(1)
            #     skiper_concate = skiper_concate.repeat(1, self.num_samples, 1, 1, 1, 1)
            #     skiper_concate = skiper_concate.view(B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d)
            #     p = torch.cat([p, skiper_concate], dim=1)
            p = decoder(p, train)
            # self.decoderout.append(p)

        recon_stft = p # B * self.num_samples, 1, F, T, D=2
        # mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        # mask_mag = torch.tanh(mask_mag)
        # real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        # imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        # mask_phase = torch.atan2(imag_phase, real_phase)

        # either recon magnitude and phase or recon real and imag
        # mag and phase
        # recon_mag = torch.sqrt(recon_stft[:,:,:,:,0].pow(2)+recon_stft[:,:,:,:,1].pow(2)) # B*numsamples, 1, freq, time, D
        # recon_phase = torch.arctan2(recon_stft[:,:,:,:,1], recon_stft[:,:,:,:,0]) # may be nan TODO: add protection
        # predict = recon_mag * torch.exp(1j * (recon_phase))        
        # predict = torch.squeeze(predict, 1)
        # recon_sig = self.istft(predict) # B * numsamples, time_len

        # real and imag
        if self.recon_type == 'real_imag':
            predict = torch.squeeze(recon_stft, 1)# B*numsamples, freq, time
            if self.datanorm:
                predict = self.data_std * predict + self.data_mean
            predict = torch.complex(predict[:,:,:,0],predict[:,:,:,1]) #B*numsamples, 1, freq, time
            recon_sig = self.istft(predict) # B * numsamples, time_len
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])

        if self.recon_type == 'mask':
            mask = recon_stft
            mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
            mask_mag = torch.tanh(mask_mag)
            real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
            imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            b,freq, time, dim = stft_x.shape
            stft_x = stft_x.unsqueeze(1)
            stft_x = stft_x.repeat(1,self.num_samples, 1, 1, 1)
            stft_x = stft_x.view(b * self.num_samples, freq, time, dim)
            stft_x = stft_x.unsqueeze(1)
            input_mag = torch.sqrt(stft_x[:,:,:,:,0].pow(2)+stft_x[:,:,:,:,1].pow(2))
            input_phase = torch.arctan2(stft_x[:,:,:,:,1], stft_x[:,:,:,:,0])
            predict = input_mag * mask_mag * torch.exp(1j * (input_phase + mask_phase))         
            predict = torch.squeeze(predict, 1)
            if self.datanorm:
                predict = torch.view_as_real(predict) # b,f,t,2
                predict = self.data_std * predict + self.data_mean
                predict = torch.complex(predict[...,0],predict[...,1])
            recon_sig = self.istft(predict)
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])
        return recon_sig, predict
    


class nsvae_dccrn_encoder_original(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        if self.latent_num == 1:
            lstm_outdim = int(3 * self.zdim)
        elif self.latent_num == 2:
            lstm_outdim = int(6 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 (or 6)--> miu, sigma, delta for speech or noise or both
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps

        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 or 6 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 or 6 * zdim, D
        
        if self.latent_num == 1:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:, :]

            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:3*self.zdim, :]

            miu_noise = lstm_[:, :, 3*self.zdim:4*self.zdim, :]
            log_sigma_noise = lstm_[:, :, 4*self.zdim:5*self.zdim, :]
            delta_noise = lstm_[:, :, 5*self.zdim:6*self.zdim, :]
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples)


        # z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x

class nsvae_dccrn_encoder_original_fc_latent(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        if self.latent_num == 1:
            self.speech_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_delta = ComplexDense(dense_inputsize, zdim)
        elif self.latent_num == 2:
            self.speech_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_delta = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_delta = ComplexDense(dense_inputsize, zdim)
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim

        lstm_outdim = self.zdim

        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 (or 6)--> miu, sigma, delta for speech or noise or both
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(torch.clamp(log_sigma[:,:,:,0], -13, 13))
        # real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)

        # all of denominators and nominators are of size B, T, H
        sqrt_arg = torch.clamp(2 * (real_sigma + real_delta), min=self.epsilon)
        denominator = torch.sqrt(sqrt_arg)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        sqrt_arg = torch.clamp(real_sigma.pow(2)-abs_delta.pow(2), min=self.epsilon)
        imag_scale_y = torch.sqrt(sqrt_arg) / (denominator)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 or 6 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, zdim, D
        
        if self.latent_num == 1:
            miu_speech = self.speech_dense_mean(lstm_)
            log_sigma_speech = self.speech_dense_logvar(lstm_)
            delta_speech = self.speech_dense_delta(lstm_)

            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = self.speech_dense_mean(lstm_)
            log_sigma_speech = self.speech_dense_logvar(lstm_)
            delta_speech = self.speech_dense_delta(lstm_)

            miu_noise = self.noise_dense_mean(lstm_)
            log_sigma_noise = self.noise_dense_logvar(lstm_)
            delta_noise = self.noise_dense_delta(lstm_)
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples)


        # z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x



class nsvae_dccrn_encoder_double_channel(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1): # index is index of encoder
            if index == 0:
                num_in_channel = en_channels[index]
            else:
                num_in_channel = 2 * en_channels[index]
            
            num_out_channel = 2 * en_channels[index + 1]


            chw_num = tuple(tt * 2 for tt in encoder_chw[index])
            model = Encoder(
                in_channel=num_in_channel, out_channel=num_out_channel,
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=chw_num, causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        if self.latent_num == 1:
            lstm_outdim = int(3 * self.zdim)
        elif self.latent_num == 2:
            lstm_outdim = int(6 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            lstm_inputdim = 2 * lstm_dims[index]
            model = ComplexLSTM(input_size=lstm_inputdim, hidden_size=lstm_outdim, # hidden_size=zdim * 3 (or 6)--> miu, sigma, delta for speech or noise or both
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps

        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 or 6 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 or 6 * zdim, D
        
        if self.latent_num == 1:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:, :]

            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:3*self.zdim, :]

            miu_noise = lstm_[:, :, 3*self.zdim:4*self.zdim, :]
            log_sigma_noise = lstm_[:, :, 4*self.zdim:5*self.zdim, :]
            delta_noise = lstm_[:, :, 5*self.zdim:6*self.zdim, :]
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples)


        # z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x


class nsvae_dccrn_encoder_adapt_channel(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num, skip_to_use):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for idx, c_num in enumerate(en_channels[1:]):
            if (len(en_channels) -2 - idx) in skip_to_use:
                en_channels[idx + 1] = c_num * 2
                encoder_chw[idx] = tuple(tt * 2 for tt in encoder_chw[idx])
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        for index in range(len(en_channels) - 1): # index is index of encoder
            num_in_channel = en_channels[index]           
            num_out_channel = en_channels[index + 1]
            model = Encoder(
                in_channel=num_in_channel, out_channel=num_out_channel,
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index], causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        if self.latent_num == 1:
            lstm_outdim = int(3 * self.zdim)
        elif self.latent_num == 2:
            lstm_outdim = int(6 * self.zdim)
        self.num_samples = num_samples

        for index in range(len(net_params["lstm_dim"]) - 1):
            if 0 in skip_to_use:
                lstm_inputdim = 2 * lstm_dims[index]
            else:
                lstm_inputdim = lstm_dims[index]
            model = ComplexLSTM(input_size=lstm_inputdim, hidden_size=lstm_outdim, # hidden_size=zdim * 3 (or 6)--> miu, sigma, delta for speech or noise or both
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps

        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 or 6 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 or 6 * zdim, D
        
        if self.latent_num == 1:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:, :]

            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:3*self.zdim, :]

            miu_noise = lstm_[:, :, 3*self.zdim:4*self.zdim, :]
            log_sigma_noise = lstm_[:, :, 4*self.zdim:5*self.zdim, :]
            delta_noise = lstm_[:, :, 5*self.zdim:6*self.zdim, :]
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples)


        # z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2
        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x     





class pvae_dccrn_encoder_prob_skip(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples):
        super().__init__()
        self.device = device
        self.causal = causal
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = int(3 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D
        
        miu = lstm_[:, :, 0:self.zdim, :]
        log_sigma = lstm_[:, :, self.zdim:2*self.zdim, :]
        delta = lstm_[:, :, 2*self.zdim:, :]

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2

        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x

class pvae_dccrn_decoder_prob_skip(nn.Module):
    def __init__(self, net_params, causal, device, num_samples, zdim, n_fft, hop_len, win_length, recon_type, skip_to_use, skip_prob):
        super().__init__()
        self.device = device
        self.causal = causal
        self.num_samples = num_samples
        self.zdim = zdim
        self.recon_type = recon_type
        self.skip_to_use = skip_to_use
        self.skip_prob = skip_prob
        if self.skip_prob == 1:
            self.zero_flag = True
        elif self.skip_prob == 2:
            self.zero_flag = False
        self.dense = ComplexDense(zdim, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        decoder_chw = net_params["decoder_chw"]

        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            if index in skip_to_use:
                in_decoder_channel = de_channels[index] + en_channels[len(en_channels) - 1 - index]
            else:
                in_decoder_channel = de_channels[index]
            model = Decoder(
                in_channel= in_decoder_channel,
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index], causal=self.causal
            )
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.decoders = nn.ModuleList(self.decoders)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)

    def forward(self, stft_x, z, skiper, C, F, train=True):

        B_numsamples, T, zdim, D = z.shape
        lstm_out = z.reshape(B_numsamples * T, -1, D) # B * self.num_samples * T, zdim, D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B_numsamples, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B * self.num_samples, C, F, T, D
        whether_sc_prob = torch.rand(1)
        if train:
            if whether_sc_prob[0] < 0.5:
                sc_flag = True
            else:
                sc_flag = False
        else:
            sc_flag = True
        
        for index, decoder in enumerate(self.decoders):
            # if index % 2 != 0:
            if sc_flag:
                if index in self.skip_to_use:
                    skiper_concate = skiper[len(skiper) - index - 1]
                    tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
                    skiper_concate = skiper_concate.unsqueeze(1)
                    skiper_concate = skiper_concate.repeat(1, self.num_samples, 1, 1, 1, 1)
                    skiper_concate = skiper_concate.view(B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d)
                    p = torch.cat([p, skiper_concate], dim=1)
            else:
                if index in self.skip_to_use:
                    if self.zero_flag:
                        skiper_concate = skiper[len(skiper) - index - 1]
                        tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
                        skiper_concate = torch.zeros((B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d), device=self.device)
                    else:
                        skiper_concate = p
                    p = torch.cat([p, skiper_concate], dim=1)
            p = decoder(p, train)

        recon_stft = p # B * self.num_samples, 1, F, T, D=2
        # mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        # mask_mag = torch.tanh(mask_mag)
        # real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        # imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        # mask_phase = torch.atan2(imag_phase, real_phase)

        # either recon magnitude and phase or recon real and imag
        # mag and phase
        # recon_mag = torch.sqrt(recon_stft[:,:,:,:,0].pow(2)+recon_stft[:,:,:,:,1].pow(2)) # B*numsamples, 1, freq, time, D
        # recon_phase = torch.arctan2(recon_stft[:,:,:,:,1], recon_stft[:,:,:,:,0]) # may be nan TODO: add protection
        # predict = recon_mag * torch.exp(1j * (recon_phase))        
        # predict = torch.squeeze(predict, 1)
        # recon_sig = self.istft(predict) # B * numsamples, time_len

        # real and imag
        if self.recon_type == 'real_imag':
            predict = torch.complex(recon_stft[:,:,:,:,0],recon_stft[:,:,:,:,1]) #B*numsamples, 1, freq, time
            predict = torch.squeeze(predict, 1)# B*numsamples, freq, time
            recon_sig = self.istft(predict) # B * numsamples, time_len

        return recon_sig, predict
    








class pvae_dccrn_encoder_skip_prepare(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples):
        super().__init__()
        self.device = device
        self.causal = causal
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = int(3 * self.zdim)
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D
        
        miu = lstm_[:, :, 0:self.zdim, :]
        log_sigma = lstm_[:, :, self.zdim:2*self.zdim, :]
        delta = lstm_[:, :, 2*self.zdim:, :]

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2

        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x
    

class pvae_dccrn_encoder_skip_prepare_fc_latent(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples):
        super().__init__()
        self.device = device
        self.causal = causal
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense_mean = ComplexDense(dense_inputsize, zdim)
        self.dense_logvar = ComplexDense(dense_inputsize, zdim)
        self.dense_delta = ComplexDense(dense_inputsize, zdim)
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = self.zdim
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)

        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(torch.clamp(log_sigma[:,:,:,0], -13, 13))
        # real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)

        # all of denominators and nominators are of size B, T, H
        sqrt_arg = torch.clamp(2 * (real_sigma + real_delta), min=self.epsilon)
        denominator = torch.sqrt(sqrt_arg)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        sqrt_arg = torch.clamp(real_sigma.pow(2)-abs_delta.pow(2), min=self.epsilon)
        imag_scale_y = torch.sqrt(sqrt_arg) / (denominator)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3 * zdim, D
        
        miu = self.dense_mean(lstm_)
        log_sigma = self.dense_logvar(lstm_)
        delta = self.dense_delta(lstm_)

        z = self.reparameterization(miu, log_sigma, delta, self.num_samples) # B*numsamples, T, zdim, D=2

        # z = miu.unsqueeze(1)
        # z = z.repeat(1, self.num_samples, 1, 1, 1)
        # z = z.view(B*self.num_samples, T, self.zdim, D)

        return z, miu, log_sigma, delta, skiper, C, F, stft_x

class pvae_dccrn_decoder_skip_prepare(nn.Module):
    def __init__(self, net_params, causal, device, num_samples, zdim, n_fft, hop_len, win_length, recon_type, skip_to_use):
        super().__init__()
        self.device = device
        self.causal = causal
        self.num_samples = num_samples
        self.zdim = zdim
        self.recon_type = recon_type
        self.skip_to_use = skip_to_use
        self.dense = ComplexDense(zdim, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        decoder_chw = net_params["decoder_chw"]

        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            if index in skip_to_use:
                in_decoder_channel = de_channels[index] + en_channels[len(en_channels) - 1 - index]
            else:
                in_decoder_channel = de_channels[index]
            model = Decoder(
                in_channel= in_decoder_channel,
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index], causal=self.causal
            )
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.decoders = nn.ModuleList(self.decoders)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)

    def forward(self, stft_x, z, skiper, C, F, train=True):

        B_numsamples, T, zdim, D = z.shape
        lstm_out = z.reshape(B_numsamples * T, -1, D) # B * self.num_samples * T, zdim, D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B_numsamples, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B * self.num_samples, C, F, T, D

        self.decoder_outputs = []
        
        for index, decoder in enumerate(self.decoders):
            if index in self.skip_to_use:
                skiper_concate = skiper[len(skiper) - index - 1]
                tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
                skiper_concate = torch.zeros((B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d), device=self.device)
                p = torch.cat([p, skiper_concate], dim=1)
            p = decoder(p, train)
            self.decoder_outputs.append(p)

        recon_stft = p # B * self.num_samples, 1, F, T, D=2
        # mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        # mask_mag = torch.tanh(mask_mag)
        # real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        # imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        # mask_phase = torch.atan2(imag_phase, real_phase)

        # either recon magnitude and phase or recon real and imag
        # mag and phase
        # recon_mag = torch.sqrt(recon_stft[:,:,:,:,0].pow(2)+recon_stft[:,:,:,:,1].pow(2)) # B*numsamples, 1, freq, time, D
        # recon_phase = torch.arctan2(recon_stft[:,:,:,:,1], recon_stft[:,:,:,:,0]) # may be nan TODO: add protection
        # predict = recon_mag * torch.exp(1j * (recon_phase))        
        # predict = torch.squeeze(predict, 1)
        # recon_sig = self.istft(predict) # B * numsamples, time_len

        # real and imag
        if self.recon_type == 'real_imag':
            predict = torch.complex(recon_stft[:,:,:,:,0],recon_stft[:,:,:,:,1]) #B*numsamples, 1, freq, time
            predict = torch.squeeze(predict, 1)# B*numsamples, freq, time
            recon_sig = self.istft(predict) # B * numsamples, time_len

        return recon_sig, predict
    







class nsvae_pvae_dccrn_encoder_twophase(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        self.dense = ComplexDense(dense_inputsize, net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        if self.latent_num == 1:
            lstm_outdim = int(3 * self.zdim)
        elif self.latent_num == 2:
            lstm_outdim = int(6 * self.zdim)
            
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)


        # all of denominators and nominators are of size B, T, H
        denominator = torch.sqrt(2 * (real_sigma + real_delta) + self.epsilon)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator + self.epsilon)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta.pow(2) + self.epsilon) / (denominator + self.epsilon)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator + self.epsilon)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3or6 * zdim, D

        if self.latent_num == 1:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:, :]
            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples) # B*numsamples, T, zdim, D=2
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = lstm_[:, :, 0:self.zdim, :]
            log_sigma_speech = lstm_[:, :, self.zdim:2*self.zdim, :]
            delta_speech = lstm_[:, :, 2*self.zdim:3*self.zdim, :]
            miu_noise = lstm_[:, :, 3*self.zdim:4*self.zdim, :]
            log_sigma_noise = lstm_[:, :, 4*self.zdim:5*self.zdim, :]
            delta_noise = lstm_[:, :, 5*self.zdim:6*self.zdim, :]            
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples) # B*numsamples, T, zdim, D=2
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples) # B*numsamples, T, zdim, D=2

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x

# In model/pvae_module.py
class dis_Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None, causal=False):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        if not causal:
            self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        else:
            self.conv = causal_complex_conv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                        stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2], dis_cbn=True)
        self.prelu = nn.PReLU()
    def check_and_log_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
            # self.detect_anormal = False
    def forward(self, x, train):
        x = self.conv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x
class distinguisher(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length):
        super().__init__()
        self.device = device
        self.causal = causal
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        for index in range(len(en_channels) - 1):
            model = dis_Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = nn.LSTM(input_size=lstm_dims[index]*2, hidden_size=1, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6


    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # B, T, CF, D
        lstm_ = lstm_.reshape(T,B,-1) # B,T,CFD

        for index, lstm in enumerate(self.lstms):
            lstm_, _ = lstm(lstm_) # B, T, 1

        lstm_ = lstm_.permute(1, 0, 2) # B, T, 1
        # lstm_ = torch.sigmoid(lstm_)
        # lstm_ = self.linear(lstm_)


        return lstm_

class nsvae_pvae_dccrn_encoder_twophase_fc_latent(nn.Module):
    def __init__(self, net_params, causal, device, zdim, n_fft, hop_len, win_length, num_samples, latent_num):
        super().__init__()
        self.device = device
        self.causal = causal
        self.latent_num = latent_num
        self.encoders = []
        self.lstms = []
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        dense_inputsize = zdim
        if self.latent_num == 1:
            self.speech_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_delta = ComplexDense(dense_inputsize, zdim)
        elif self.latent_num == 2:
            self.speech_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.speech_dense_delta = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_mean = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_logvar = ComplexDense(dense_inputsize, zdim)
            self.noise_dense_delta = ComplexDense(dense_inputsize, zdim)
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index],causal=self.causal
            )
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        self.zdim = zdim
        lstm_outdim = self.zdim
            
        self.num_samples = num_samples
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_outdim, # hidden_size=zdim * 3 --> miu, sigma, delta
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)


        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)

        self.epsilon = 1e-6

    def reparameterization(self, miu, log_sigma, delta, num_samples):
        real_miu = miu[:,:,:,0] # B, T, H
        imag_miu = miu[:,:,:,1]
        real_sigma = torch.exp(torch.clamp(log_sigma[:,:,:,0], -13, 13))
        # real_sigma = torch.exp(log_sigma[:,:,:,0])
        real_delta = delta[:,:,:,0]
        imag_delta = delta[:,:,:,1]

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)
        temp = real_sigma * 0.99 / (abs_delta + self.epsilon)

        real_delta = torch.where(abs_delta >= (real_sigma-1e-3), real_delta * temp, real_delta)
        imag_delta = torch.where(abs_delta >= (real_sigma-1e-3), imag_delta * temp, imag_delta)

        # abs_delta_square = real_delta.pow(2) + imag_delta.pow(2)
        abs_delta = torch.sqrt(real_delta.pow(2) + imag_delta.pow(2) + self.epsilon)

        # all of denominators and nominators are of size B, T, H
        sqrt_arg = torch.clamp(2 * (real_sigma + real_delta), min=self.epsilon)
        denominator = torch.sqrt(sqrt_arg)

        real_nominator = real_sigma + real_delta

        imag_scale_x = imag_delta / (denominator)


        # imag_scale_y = torch.sqrt(real_sigma.pow(2)-abs_delta_square) / (denominator + self.epsilon)
        sqrt_arg = torch.clamp(real_sigma.pow(2)-abs_delta.pow(2), min=self.epsilon)
        imag_scale_y = torch.sqrt(sqrt_arg) / (denominator)


        denominator = denominator.unsqueeze(1) # B, 1, T, H
        real_nominator = real_nominator.unsqueeze(1)# B, 1, T, H
        imag_scale_x = imag_scale_x.unsqueeze(1) # B, 1, T, H
        imag_scale_y = imag_scale_y.unsqueeze(1)# B, 1, T, H
        real_miu = real_miu.unsqueeze(1)# B, 1, T, H
        imag_miu = imag_miu.unsqueeze(1)# B, 1, T, H

        # resample

        real_miu = real_miu.repeat(1, num_samples, 1, 1) # B, num_samples, T, H
        imag_miu = imag_miu.repeat(1, num_samples, 1, 1)

        real_eps = torch.randn_like(real_miu)
        imag_eps = torch.randn_like(imag_miu)

        real_z = real_miu + (real_nominator / (denominator)) * real_eps # B, num_samples, T, H
        
        imag_z = imag_miu + imag_scale_x * real_eps + imag_scale_y * imag_eps
        B, _, T, H = real_z.shape
        real_z = real_z.view(B * num_samples, T, H)
        imag_z = imag_z.view(B * num_samples, T, H)

        output = torch.stack((real_z, imag_z), dim=-1)

        return output

    def forward(self, x, train=True):
        stft_x = self.stft(x) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        x = torch.unsqueeze(stft_x, 1)
        skiper = []
        for index, encoder in enumerate(self.encoders):
            # skiper.append(x)
            x = encoder(x, train)
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D) # B, CF, T, D
        lstm_ = lstm_.permute(2, 0, 1, 3) # T, B, CF, D
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_) # T, B, 3 * zdim, D

        lstm_ = lstm_.permute(1, 0, 2, 3) # B, T, 3or6 * zdim, D

        if self.latent_num == 1:
            miu_speech = self.speech_dense_mean(lstm_)
            log_sigma_speech = self.speech_dense_logvar(lstm_)
            delta_speech = self.speech_dense_delta(lstm_)

            miu_noise = None
            log_sigma_noise = None
            delta_noise = None
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = None
        elif self.latent_num == 2:
            miu_speech = self.speech_dense_mean(lstm_)
            log_sigma_speech = self.speech_dense_logvar(lstm_)
            delta_speech = self.speech_dense_delta(lstm_)
            
            miu_noise = self.noise_dense_mean(lstm_)
            log_sigma_noise = self.noise_dense_logvar(lstm_)
            delta_noise = self.noise_dense_delta(lstm_)
            z_speech = self.reparameterization(miu_speech, log_sigma_speech, delta_speech, self.num_samples)
            z_noise = self.reparameterization(miu_noise, log_sigma_noise, delta_noise, self.num_samples)

        return z_speech, miu_speech, log_sigma_speech, delta_speech, z_noise, miu_noise, log_sigma_noise, delta_noise, skiper, C, F, stft_x



class nsvae_pvae_dccrn_decoder_twophase(nn.Module):
    def __init__(self, net_params, causal, device, num_samples, zdim, n_fft, hop_len, win_length, recon_type, use_sc, skip_to_use, resynthesis):
        super().__init__()
        self.device = device
        self.causal = causal
        self.num_samples = num_samples
        self.zdim = zdim
        self.recon_type = recon_type
        self.skip_to_use = skip_to_use
        self.dense = ComplexDense(zdim, net_params["dense"][1])
        self.decoders = []
        self.use_sc = use_sc
        self.resynthesis = resynthesis
        # init encoders
        en_channels = net_params["encoder_channels"]
        decoder_chw = net_params["decoder_chw"]

        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            if self.use_sc:
                if index in skip_to_use:
                    in_decoder_channel = de_channels[index] + en_channels[len(en_channels) - 1 - index]
                else:
                    in_decoder_channel = de_channels[index]
            else:
                in_decoder_channel = de_channels[index]
            model = Decoder(
                in_channel= in_decoder_channel,
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index], causal=self.causal
            )
            # self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.decoders = nn.ModuleList(self.decoders)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)

    def forward(self, stft_x, z, skiper, C, F, train=True, pad='zero'):

        B_numsamples, T, zdim, D = z.shape
        lstm_out = z.reshape(B_numsamples * T, -1, D) # B * self.num_samples * T, zdim, D
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B_numsamples, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4) # B * self.num_samples, C, F, T, D
        
        for index, decoder in enumerate(self.decoders):
            if self.use_sc:
                if index in self.skip_to_use:
                    skiper_concate = skiper[len(skiper) - index - 1]
                    tmp_b, tmp_c, tmp_f, tmp_t, tmp_d = skiper_concate.shape
                    if pad == 'zero':
                        skiper_concate = torch.zeros((B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d), device=self.device)
                    elif pad == 'sig':
                        skiper_concate = skiper_concate.unsqueeze(1)
                        skiper_concate = skiper_concate.repeat(1, self.num_samples, 1, 1, 1, 1)
                        skiper_concate = skiper_concate.view(B_numsamples, tmp_c, tmp_f, tmp_t, tmp_d)                    
                    p = torch.cat([p, skiper_concate], dim=1)
            p = decoder(p, train)

        recon_stft = p # B * self.num_samples, 1, F, T, D=2
        # mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        # mask_mag = torch.tanh(mask_mag)
        # real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        # imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        # mask_phase = torch.atan2(imag_phase, real_phase)

        # either recon magnitude and phase or recon real and imag
        # mag and phase
        # recon_mag = torch.sqrt(recon_stft[:,:,:,:,0].pow(2)+recon_stft[:,:,:,:,1].pow(2)) # B*numsamples, 1, freq, time, D
        # recon_phase = torch.arctan2(recon_stft[:,:,:,:,1], recon_stft[:,:,:,:,0]) # may be nan TODO: add protection
        # predict = recon_mag * torch.exp(1j * (recon_phase))        
        # predict = torch.squeeze(predict, 1)
        # recon_sig = self.istft(predict) # B * numsamples, time_len

        # real and imag
        if self.recon_type == 'real_imag':
            predict = torch.complex(recon_stft[:,:,:,:,0],recon_stft[:,:,:,:,1]) #B*numsamples, 1, freq, time
            predict = torch.squeeze(predict, 1)# B*numsamples, freq, time
            recon_sig = self.istft(predict) # B * numsamples, time_len
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])

        if self.recon_type == 'mask':
            mask = recon_stft
            mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
            mask_mag = torch.tanh(mask_mag)
            real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
            imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            b,freq, time, dim = stft_x.shape
            stft_x = stft_x.unsqueeze(1)
            stft_x = stft_x.repeat(1,self.num_samples, 1, 1, 1)
            stft_x = stft_x.view(b * self.num_samples, freq, time, dim)
            stft_x = stft_x.unsqueeze(1)
            input_mag = torch.sqrt(stft_x[:,:,:,:,0].pow(2)+stft_x[:,:,:,:,1].pow(2))
            input_phase = torch.arctan2(stft_x[:,:,:,:,1], stft_x[:,:,:,:,0])
            predict = input_mag * mask_mag * torch.exp(1j * (input_phase + mask_phase))         
            predict = torch.squeeze(predict, 1)
            # if self.datanorm:
            #     predict = torch.view_as_real(predict) # b,f,t,2
            #     predict = self.data_std * predict + self.data_mean
            #     predict = torch.complex(predict[...,0],predict[...,1])
            recon_sig = self.istft(predict)
            if self.resynthesis:
                resyn_stft = self.stft(recon_sig)
                predict = torch.complex(resyn_stft[...,0],resyn_stft[...,1])

        return recon_sig, predict