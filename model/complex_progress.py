# coding: utf-8
# Author：WangTianRui
# Date ：2020/8/18 9:43

import torch
import torch.nn as nn

class causal_complex_conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        real = real[:, :, :, :-1]
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        imaginary = imaginary[:, :, :, :-1]
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm_re = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        # batch_size = x.size(1)
        # h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # real_real, (h_real, c_real) = self.lstm_re(x[..., 0], (h_real, c_real))
        # imag_imag, (h_imag, c_imag) = self.lstm_im(x[..., 1], (h_imag, c_imag))
        real_real, _ = self.lstm_re(x[..., 0])
        real_imag, _ = self.lstm_im(x[..., 0])
        imag_imag, _ = self.lstm_im(x[..., 1])
        imag_real, _ = self.lstm_re(x[..., 1])
        # real_imag, _ = self.lstm_im(x[..., 0])
        real = real_real - imag_imag
        # h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        # imag_real, _ = self.lstm_re(x[..., 1])
        # real_imag, _ = self.lstm_im(x[..., 0])
        imaginary = imag_real + real_imag
        # print(f"real range: {real.min().item():.4f}~{real.max().item():.4f}")
        # print(f"imag range: {imaginary.min().item():.4f}~{imaginary.max().item():.4f}")
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear_read = nn.Linear(in_channel, out_channel)
        self.linear_imag = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        real = self.linear_read(real)
        imag = self.linear_imag(imag)
        out = torch.stack((real, imag), dim=-1)
        return out


class ComplexBatchNormal(nn.Module):
    def __init__(self, C, H, W, momentum=0.9,dis_cbn=False):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.ones(C), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(C), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.ones(C), requires_grad=True)
        self.beta_r = nn.Parameter(torch.zeros(C), requires_grad=True)
        self.beta_i = nn.Parameter(torch.zeros(C), requires_grad=True)
        self.epsilon = 1e-5
        # self.running_mean_real = None
        # self.running_mean_imag = None
        # self.Vrr = None
        # self.Vri = None
        # self.Vii = None

        self.register_buffer('running_mean_real',  torch.zeros(1,C,1,1))
        self.register_buffer('running_mean_imag',  torch.zeros(1,C,1,1))
        self.register_buffer('Vrr', torch.ones(1,C,1,1))
        self.register_buffer('Vri', torch.zeros(1,C,1,1))
        self.register_buffer('Vii', torch.ones(1,C,1,1))

        self.init_flag = True
        self.detect_anormal = True
        self.dis_cbn = dis_cbn


    def check_and_log_nan(self, tensor, name, input):
        if torch.isnan(tensor).any() and self.detect_anormal:
            print(f"NaN detected in {name}")
            print(input)
            # print(f"Tensor contents:\n{tensor}")
            # self.detect_anormal = False
            raise RuntimeError(f"NaN detected in {name}")

    def forward(self, x, train=True):
        B, C, H, W, D = x.size()
        real = x[..., 0]
        imaginary = x[..., 1]
        if train:
            mu_real = torch.mean(real, dim=(0,2,3), keepdim=True) # 1,c,1,1
            mu_imag = torch.mean(imaginary, dim=(0,2,3), keepdim=True) # 1,c,1,1

            broadcast_mu_real = mu_real 
            broadcast_mu_imag = mu_imag 

            real_centred = real - broadcast_mu_real # b,c,f,t
            imag_centred = imaginary - broadcast_mu_imag # b,c,f,t

            Vrr = torch.mean(real_centred * real_centred, (0,2,3), keepdim=True) + self.epsilon # 1,c,1,1
            Vii = torch.mean(imag_centred * imag_centred, (0,2,3), keepdim=True) + self.epsilon # 1,c,1,1
            Vri = torch.mean(real_centred * imag_centred, (0,2,3), keepdim=True) # 1,c,1,1
            if self.init_flag:
                self.running_mean_real.copy_(mu_real)
                self.running_mean_imag.copy_(mu_imag)
                self.Vrr.copy_(Vrr)
                self.Vri.copy_(Vri)
                self.Vii.copy_(Vii)
                if not self.dis_cbn:
                    self.init_flag = False
            else:
                # momentum
                with torch.no_grad():
                    self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                    self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                    self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                    self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                    self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, C)
        else:
            broadcast_mu_real = self.running_mean_real
            broadcast_mu_imag = self.running_mean_imag
            real_centred = real - broadcast_mu_real # b,c,f,t
            imag_centred = imaginary - broadcast_mu_imag
            return self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, C)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, C):
        tau = Vrr + Vii # 1,c,1,1
        # delta = (Vrr * Vii) - (Vri ** 2) # 1,c,1,1
        # s = torch.sqrt(delta) # 1,c,1,1

        # t = torch.sqrt(tau + 2 * s) # 1,c,1,1
        # inverse_st = 1.0 / (s * t) # 1,c,1,1:

        # self.check_and_log_nan(s, "s")
        # self.check_and_log_nan(t, "t")
        # self.check_and_log_nan(inverse_st, "inverse_st")
        
        delta = (Vrr * Vii) - (Vri ** 2) + self.epsilon # 1,c,1,1
        delta = torch.clamp(delta, min=1e-8)
        s = torch.sqrt(delta) # 1,c,1,1
        # self.check_and_log_nan(s, "s", delta)
        t = torch.sqrt(tau + 2 * s + self.epsilon) # 1,c,1,1
        # self.check_and_log_nan(t, "t", input_t)
        inverse_st = 1.0 / (s * t + self.epsilon)
        # self.check_and_log_nan(inverse_st, "inverse_st", input_inverse_st)

        Wrr = (Vii + s) * inverse_st # 1,c,1,1
        Wii = ((Vrr + s) * inverse_st) # 1,c,1,1
        Wri = (-Vri * inverse_st) # 1,c,1,1

        # n_real = Wrr * real_centred + Wri * imag_centred
        # n_imag = Wii * imag_centred + Wri * real_centred

        broadcast_gamma_rr = self.gamma_rr.view(1, C, 1, 1)
        broadcast_gamma_ri = self.gamma_ri.view(1, C, 1, 1)
        broadcast_gamma_ii = self.gamma_ii.view(1, C, 1, 1)
        broadcast_beta_r = self.beta_r.view(1, C, 1, 1)
        broadcast_beta_i = self.beta_i.view(1, C, 1, 1)

        Zrr = (broadcast_gamma_rr * Wrr) + (broadcast_gamma_ri * Wri)
        Zri = (broadcast_gamma_rr * Wri) + (broadcast_gamma_ri * Wii)
        Zir = (broadcast_gamma_ri * Wrr) + (broadcast_gamma_ii * Wri)
        Zii = (broadcast_gamma_ri * Wri) + (broadcast_gamma_ii * Wii)

        bn_real = Zrr * real_centred + Zri * imag_centred + broadcast_beta_r
        bn_imag = Zir * real_centred + Zii * imag_centred + broadcast_beta_i
        return torch.stack((bn_real, bn_imag), dim=-1)


def init_get(kind):
    if kind == "sqrt_init":
        return sqrt_init
    else:
        return torch.zeros


def sqrt_init(shape):
    return (1 / torch.sqrt(torch.tensor(2))) * torch.ones(shape)

class causal_ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        real = real[:, :, :, :-1]
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        imaginary = imaginary[:, :, :, :-1]
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output