# coding: utf-8
# Author：WangTianRui
# Date ：2020/9/30 10:55


from model.complex_progress import *
# import torchaudio_contrib as audio_nn
# from utils import *


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, device):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)
        self.window = self.window.to(device)
        # self.stft = torch.stft(n_fft=self.n_fft, hop_length=self.hop_length, win_length=win_length, window=torch.hann_window(win_length),return_complex=False)

    def forward(self, signal):
        with torch.no_grad():
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
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
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
        # self.check_and_log_nan(x,'encoder conv out')
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()
    def check_and_log_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
    def forward(self, x, train=True):
        x = self.transconv(x)
        # self.check_and_log_nan(x, "decoder transconv output")
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class DCCRN(nn.Module):
    def __init__(self, net_params, device):
        super().__init__()
        self.device = device
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params["dense"][0], net_params["dense"][1])
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
                chw=encoder_chw[index]
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
            # if index % 2 != 0:
            in_decoder_channel = de_channels[index] + en_channels[len(self.encoders) - index]
            # else:
            #     in_decoder_channel = de_channels[index]
            model = Decoder(
                in_channel= in_decoder_channel,
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index]
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
            # self.check_and_log_nan(x, "encoder out x")
            skiper.append(x)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D)
        lstm_ = lstm_.permute(2, 0, 1, 3)
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_)
            # self.check_and_log_nan(lstm_, "lstm_")
        lstm_ = lstm_.permute(1, 0, 2, 3)
        lstm_out = lstm_.reshape(B * T, -1, D)
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4)
        for index, decoder in enumerate(self.decoders):
            p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
            p = decoder(p, train)

            # self.check_and_log_nan(p, "decoder out p")
            # p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
        # mask = torch.tanh(self.linear(p))
        mask = p
        mask_mag = torch.sqrt(mask[:,:,:,:,0].pow(2)+mask[:,:,:,:,1].pow(2))
        mask_mag = torch.tanh(mask_mag)
        real_phase = mask[:,:,:,:,0]/(mask_mag + 1e-8)
        imag_phase = mask[:,:,:,:,1]/(mask_mag + 1e-8)
        mask_phase = torch.atan2(imag_phase, real_phase)

        # self.check_and_log_nan(mask_mag, "mask_mag")
        # self.check_and_log_nan(mask_phase, "mask_phase", real_phase, imag_phase)
        return mask_mag, mask_phase


class DCCRN_(nn.Module):
    def __init__(self, n_fft, hop_len, net_params, device, win_length):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_length=win_length, device=device)
        self.DCCRN = DCCRN(net_params, device=device)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length, device=device)

    def forward(self, signal, train=True):
        stft = self.stft(signal) # (batch=128, freq=257, time=1601 ~ timelen//hop + 1, 2 ~ real and imag)
        stft = torch.unsqueeze(stft, 1)
        # contains_nan = torch.isnan(stft).any()
        # print(f"STFT? {contains_nan}")
        mask_mag_predict, mask_phase_predict = self.DCCRN(stft, train=train)

        input_mag = torch.sqrt(stft[:,:,:,:,0].pow(2)+stft[:,:,:,:,1].pow(2))
        # contains_nan = torch.isnan(input_mag).any()
        # print(f"input_mag? {contains_nan}")
        input_phase = torch.arctan2(stft[:,:,:,:,1], stft[:,:,:,:,0])
        # contains_nan = torch.isnan(input_phase).any()
        # print(f"input_phase? {contains_nan}")
        predict = input_mag * mask_mag_predict * torch.exp(1j * (input_phase + mask_phase_predict))
        # contains_nan = torch.isnan(mask_mag_predict).any()
        # print(f"mask_mag_predict? {contains_nan}") 
        # contains_nan = torch.isnan(mask_phase_predict).any()
        # print(f"mask_phase_predict? {contains_nan}")          
        predict = torch.squeeze(predict, 1)
        clean = self.istft(predict)
        # contains_nan = torch.isnan(clean).any()
        # print(f"clean? {contains_nan}")  
        return clean