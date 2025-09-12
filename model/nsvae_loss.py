import torch
import numpy as np
import matplotlib.pyplot as plt

class standard_nsvae_loss_by_sampling():
    def __init__(self, alpha, w_resi, w_kl, zdim, num_samples, latent_num, nsvae_model, skipc, skip_to_use, matching):
        self.alpha = alpha
        self.w_resi = w_resi
        self.w_kl = w_kl
        # self.pi_n = torch.pi ** (zdim)
        self.epsilon = 1e-10
        self.num_samples = num_samples
        self.latent_num = latent_num
        self.skip_to_use = skip_to_use
        self.nsvae_model = nsvae_model
        self.skipc = skipc
        if nsvae_model == 'adapt' or nsvae_model == 'double':
            self.skiper_split = True
        else:
            self.skiper_split = False

        self.matching = matching

    def check_and_log_nan(self,tensor, name, input0 = None, input=None, input1=None, input2=None, input3=None):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            pos = torch.nonzero(input0 < 0, as_tuple=True)
            input = input[pos]
            input1 = input1[pos]
            input2 = input2[pos]
            input3 = input3[pos]
            print(input)
            print(input1)
            print(input2)
            print(input3)
            raise RuntimeError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"inf detected in {name}")
            print(torch.min(input))
            print(torch.max(input))
            raise RuntimeError(f"inf detected in {name}")

    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        # B, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        B, T, H = sigma.shape
        delta_real = delta[:, :, :, 0]
        delta_imag = delta[:, :, :, 1]
        z_real = z[:,:,:,0] # B * numsamples, T, H
        z_real = z_real.view(B, self.num_samples, T, H) # B, numsamples, T, H
        z_imag = z[:,:,:,1]
        z_imag = z_imag.view(B, self.num_samples, T, H)
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_imag = miu[:, :, :, 1]

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.99 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=2)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=2)
        log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        miu_real = miu_real.unsqueeze(1) # B,1,T;H
        miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part

        return log_final_prob # B; numsamples, T





    def cal_kl(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

        log_final_prob1 = self.cal_gaussian_prob(miu1, log_sigma1, delta1, z1) # B, numsamples, T
        log_final_prob2 = self.cal_gaussian_prob(miu2, log_sigma2, delta2, z1)
        #estimated kl
        # kl = torch.mean(torch.log(prob_distribution_1 / (prob_distribution_2 + self.epsilon) + self.epsilon), dim=1) #B, T
        kl = torch.mean(log_final_prob1 - log_final_prob2, dim=1)

        return kl, log_final_prob1, log_final_prob2
    
    def kl_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                        log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise):
        
        if self.latent_num == 1:
            kl_clean, log_prob_p_clean, log_prob_q_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise, log_prob_p_noise, log_prob_q_noise = self.cal_kl(miu_noisy_speech, miu_noise, log_sigma_noisy_speech, log_sigma_noise, delta_noisy_speech, delta_noise, z_noisy_speech)

            kl_loss_final = torch.mean(kl_clean) - self.alpha * torch.mean(kl_noise) # per sample, per time frame

        elif self.latent_num == 2:
            kl_clean, log_prob_p_clean, log_prob_q_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise, log_prob_p_noise, log_prob_q_noise = self.cal_kl(miu_noisy_noise, miu_noise, log_sigma_noisy_noise, log_sigma_noise, delta_noisy_noise, delta_noise, z_noisy_noise)

            kl_loss_final = torch.mean(kl_clean) + torch.mean(kl_noise) # per sample, per time frame

        return kl_loss_final, torch.mean(kl_clean), torch.mean(kl_noise)


    def residual_loss(self, skiper_clean, skiper_noise, skiper_noisy):

        if self.latent_num == 1:
            if self.skiper_split:
                final_loss_speech = 0
                final_loss_noise = 0
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                        chn = skiper_noisy[idx].shape[1]
                        connct2 = skiper_noisy[idx][:,:int(chn//2), :, :, :] # first half represents speech
                        diff = (connct - connct2).pow(2)
                        # diff = torch.sum(torch.sum(diff.pow(2), dim=3), dim=2)
                        diff = torch.mean(diff) # per sample, per channel
                        final_loss_speech = final_loss_speech + diff
                final_loss = final_loss_speech
            else:
                final_loss_speech = 0
                final_loss_noise = 0
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                        connct2 = skiper_noisy[idx]
                        diff = (connct - connct2).pow(2)
                        # diff = torch.sum(torch.sum(diff.pow(2), dim=3), dim=2)
                        diff = torch.mean(diff) # per sample, per channel
                        final_loss_speech = final_loss_speech + diff

                final_loss = final_loss_speech

        elif self.latent_num == 2:
            if self.matching == 'both':
                final_loss_speech = 0
                final_loss_noise = 0
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                        chn = skiper_noisy[idx].shape[1]
                        skiper_noisy_speech = skiper_noisy[idx][:,:int(chn//2), :, :, :]
                        skiper_noisy_noise = skiper_noisy[idx][:,int(chn//2):, :, :, :]
                        diff_speech = (connct - skiper_noisy_speech).pow(2)
                        diff_noise = (skiper_noise[idx] - skiper_noisy_noise).pow(2)
                        # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                        diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                        final_loss_speech = final_loss_speech + diff_speech

                        # diff_noise = torch.sum(torch.sum(diff_noise.pow(2), dim=3), dim=2)
                        diff_noise = torch.mean(diff_noise) # per sample, per channel, per real or imag
                        final_loss_noise = final_loss_noise + diff_noise  

                final_loss = final_loss_speech + final_loss_noise
            elif self.matching == 'speech':
                if self.skiper_split:
                    final_loss_speech = 0
                    final_loss_noise = 0
                    for idx, connct in enumerate(skiper_clean):
                        if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                            chn = skiper_noisy[idx].shape[1]
                            skiper_noisy_speech = skiper_noisy[idx][:,:int(chn//2), :, :, :]
                            
                            diff_speech = (connct - skiper_noisy_speech).pow(2)
                            
                            # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                            diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                            final_loss_speech = final_loss_speech + diff_speech
                else:
                    final_loss_speech = 0
                    final_loss_noise = 0
                    for idx, connct in enumerate(skiper_clean):
                        if (len(skiper_clean) - 1 - idx) in self.skip_to_use:

                            skiper_noisy_speech = skiper_noisy[idx]
                            
                            diff_speech = (connct - skiper_noisy_speech).pow(2)
                            
                            # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                            diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                            final_loss_speech = final_loss_speech + diff_speech


                final_loss = final_loss_speech


        return final_loss, final_loss_speech, final_loss_noise
    

    def final_nsvae_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise, 
                       skiper_clean, skiper_noise, skiper_noisy):
        

        kl_loss, kl_clean, kl_noise = self.kl_loss(miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise)

        if self.skipc == 'True' and self.w_resi != 0:
            resi_loss, resi_loss_speech, resi_loss_noise = self.residual_loss(skiper_clean, skiper_noise, skiper_noisy)

            final_loss = self.w_kl * kl_loss + self.w_resi * resi_loss
        else:
            resi_loss, resi_loss_speech, resi_loss_noise = 0, 0, 0
            final_loss = self.w_kl * kl_loss

        return final_loss,kl_loss, kl_clean, kl_noise, resi_loss, resi_loss_speech, resi_loss_noise



class standard_nsvae_loss_true_kl():
    def __init__(self, alpha, w_resi, w_kl, w_dismiu, zdim, num_samples, latent_num, nsvae_model, skipc, skip_to_use, matching):
        self.alpha = alpha
        self.w_resi = w_resi
        self.w_kl = w_kl
        self.w_dismiu = w_dismiu
        # self.pi_n = torch.pi ** (zdim)
        self.epsilon = 1e-10
        self.num_samples = num_samples
        self.latent_num = latent_num
        self.skip_to_use = skip_to_use
        self.nsvae_model = nsvae_model
        self.skipc = skipc
        if nsvae_model == 'adapt' or nsvae_model == 'double':
            self.skiper_split = True
        else:
            self.skiper_split = False

        self.matching = matching
        self.zdim = zdim

    def check_and_log_nan(self,tensor, name, input0 = None):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise RuntimeError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"inf detected in {name}")
            raise RuntimeError(f"inf detected in {name}")




    def cal_kl(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

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
        abs_delta1 = torch.sqrt(delta1_real.pow(2) + delta1_imag.pow(2) + self.epsilon)
        temp = sigma1 * 0.99 / (abs_delta1 + self.epsilon)

        delta1_real = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_real * temp, delta1_real)
        delta1_imag = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_imag * temp, delta1_imag)

        abs_delta1 = delta1_real.pow(2) + delta1_imag.pow(2)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta2 = torch.sqrt(delta2_real.pow(2) + delta2_imag.pow(2) + self.epsilon)
        temp = sigma2 * 0.99 / (abs_delta2 + self.epsilon)

        delta2_real = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_real * temp, delta2_real)
        delta2_imag = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_imag * temp, delta2_imag)

        abs_delta2 = delta2_real.pow(2) + delta2_imag.pow(2)

        # B,T,H
        log_det_c1 = torch.log(0.25 * (sigma1.pow(2) - abs_delta1) + self.epsilon)
        log_det_c2 = torch.log(0.25 * (sigma2.pow(2) - abs_delta2) + self.epsilon)

        coeff = 2 / (sigma2.pow(2) - abs_delta2 + self.epsilon)


        trace_term = sigma1 * sigma2 - delta2_real * delta1_real - delta2_imag * delta1_imag

        miu_diff_real = miu2_real - miu1_real
        miu_diff_imag = miu2_imag - miu1_imag
        quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * (trace_term + quadra_term) + log_det_c2 - log_det_c1, dim=2) - self.zdim

        return kl
    
    def kl_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                        log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise):
        
        if self.latent_num == 1:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_speech, miu_noise, log_sigma_noisy_speech, log_sigma_noise, delta_noisy_speech, delta_noise, z_noisy_speech)

            kl_loss_final = torch.mean(kl_clean) - self.alpha * torch.mean(kl_noise) # per sample, per time frame

        elif self.latent_num == 2:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_noise, miu_noise, log_sigma_noisy_noise, log_sigma_noise, delta_noisy_noise, delta_noise, z_noisy_noise)

            kl_loss_final = torch.mean(kl_clean) + self.alpha * torch.mean(kl_noise) # per sample, per time frame

        return kl_loss_final, torch.mean(kl_clean), torch.mean(kl_noise)

    def miu_dis_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise,
                    log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                    delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise,):

        speech_dis_mean_loss = torch.mean((miu_clean - miu_noisy_speech).pow(2),dim=(0,1)) # dim, 2
        speech_dis_mean_loss = torch.sqrt(torch.sum(speech_dis_mean_loss))
        noise_dis_mean_loss = torch.mean((miu_noise - miu_noisy_noise).pow(2),dim=(0,1)) # dim, 2
        noise_dis_mean_loss = torch.sqrt(torch.sum(noise_dis_mean_loss))

        loss = speech_dis_mean_loss + noise_dis_mean_loss

        return loss, speech_dis_mean_loss, noise_dis_mean_loss


    def residual_loss(self, skiper_clean, skiper_noise, skiper_noisy):

        if self.latent_num == 1:
            if self.skiper_split:
                final_loss_speech = 0
                final_loss_noise = 0
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                        chn = skiper_noisy[idx].shape[1]
                        connct2 = skiper_noisy[idx][:,:int(chn//2), :, :, :] # first half represents speech
                        diff = (connct - connct2).pow(2)
                        # diff = torch.sum(torch.sum(diff.pow(2), dim=3), dim=2)
                        diff = torch.mean(diff) # per sample, per channel
                        final_loss_speech = final_loss_speech + diff
                final_loss = final_loss_speech
            else:
                final_loss_speech = 0
                final_loss_noise = 0
                # layer_weight = [2,1,1,0.01, 0.01, 0.01]
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:

                        connct2 = skiper_noisy[idx]
                        diff = (connct - connct2).pow(2)
                        # diff = torch.sum(torch.sum(diff.pow(2), dim=3), dim=2)
                        diff = torch.mean(diff) # per sample, per channel
                        final_loss_speech = final_loss_speech + diff

                final_loss = final_loss_speech

        elif self.latent_num == 2:
            if self.matching == 'both':
                final_loss_speech = 0
                final_loss_noise = 0
                for idx, connct in enumerate(skiper_clean):
                    if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                        chn = skiper_noisy[idx].shape[1]
                        skiper_noisy_speech = skiper_noisy[idx][:,:int(chn//2), :, :, :]
                        skiper_noisy_noise = skiper_noisy[idx][:,int(chn//2):, :, :, :]
                        diff_speech = (connct - skiper_noisy_speech).pow(2)
                        diff_noise = (skiper_noise[idx] - skiper_noisy_noise).pow(2)
                        # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                        diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                        final_loss_speech = final_loss_speech + diff_speech

                        # diff_noise = torch.sum(torch.sum(diff_noise.pow(2), dim=3), dim=2)
                        diff_noise = torch.mean(diff_noise) # per sample, per channel, per real or imag
                        final_loss_noise = final_loss_noise + diff_noise  

                final_loss = final_loss_speech + final_loss_noise
            elif self.matching == 'speech':
                if self.skiper_split:
                    final_loss_speech = 0
                    final_loss_noise = 0
                    for idx, connct in enumerate(skiper_clean):
                        if (len(skiper_clean) - 1 - idx) in self.skip_to_use:
                            chn = skiper_noisy[idx].shape[1]
                            skiper_noisy_speech = skiper_noisy[idx][:,:int(chn//2), :, :, :]
                            
                            diff_speech = (connct - skiper_noisy_speech).pow(2)
                            
                            # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                            diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                            final_loss_speech = final_loss_speech + diff_speech
                else:
                    final_loss_speech = 0
                    final_loss_noise = 0
                    for idx, connct in enumerate(skiper_clean):
                        if (len(skiper_clean) - 1 - idx) in self.skip_to_use:

                            skiper_noisy_speech = skiper_noisy[idx]
                            
                            diff_speech = (connct - skiper_noisy_speech).pow(2)
                            
                            # diff_speech = torch.sum(torch.sum(diff_speech.pow(2), dim=3), dim=2)
                            diff_speech = torch.mean(diff_speech) # per sample, per channel, per real or imag
                            final_loss_speech = final_loss_speech + diff_speech


                final_loss = final_loss_speech


        return final_loss, final_loss_speech, final_loss_noise
    

    def final_nsvae_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise, 
                       skiper_clean, skiper_noise, skiper_noisy):
        

        kl_loss, kl_clean, kl_noise = self.kl_loss(miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise)

        if self.skipc == 'True' and self.w_resi != 0:
            resi_loss, resi_loss_speech, resi_loss_noise = self.residual_loss(skiper_clean, skiper_noise, skiper_noisy)
            dismiu_loss, dismiu_speech, dismiu_noise = self.miu_dis_loss(miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                                                                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                                                                        delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise)
            final_loss = self.w_kl * kl_loss + self.w_dismiu * dismiu_loss
        else:
            resi_loss, resi_loss_speech, resi_loss_noise = 0, 0, 0
            dismiu_loss, dismiu_speech, dismiu_noise = self.miu_dis_loss(miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                                                                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                                                                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise)
            final_loss = self.w_kl * kl_loss + self.w_dismiu * dismiu_loss

        return final_loss,kl_loss, kl_clean, kl_noise, dismiu_speech, dismiu_noise, resi_loss, resi_loss_speech, resi_loss_noise
class nsvae_loss_with_cvae_decoder_recon():

    def __init__(self, w_kl_noise, w_kl_speech, w_recon, recon_loss_weight, latent_num, zdim):
        self.w_kl_noise = w_kl_noise
        self.w_kl_speech = w_kl_speech
        self.recon_loss_weight = recon_loss_weight
        self.w_recon = w_recon
        self.latent_num = latent_num
        self.zdim = zdim
        self.epsilon = 1e-10

    def cal_kl(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

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
        abs_delta1 = torch.sqrt(delta1_real.pow(2) + delta1_imag.pow(2) + self.epsilon)
        temp = sigma1 * 0.99 / (abs_delta1 + self.epsilon)

        delta1_real = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_real * temp, delta1_real)
        delta1_imag = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_imag * temp, delta1_imag)

        abs_delta1 = delta1_real.pow(2) + delta1_imag.pow(2)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta2 = torch.sqrt(delta2_real.pow(2) + delta2_imag.pow(2) + self.epsilon)
        temp = sigma2 * 0.99 / (abs_delta2 + self.epsilon)

        delta2_real = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_real * temp, delta2_real)
        delta2_imag = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_imag * temp, delta2_imag)

        abs_delta2 = delta2_real.pow(2) + delta2_imag.pow(2)

        # B,T,H
        log_det_c1 = torch.log(0.25 * (sigma1.pow(2) - abs_delta1) + self.epsilon)
        log_det_c2 = torch.log(0.25 * (sigma2.pow(2) - abs_delta2) + self.epsilon)

        coeff = 2 / (sigma2.pow(2) - abs_delta2 + self.epsilon)


        trace_term = sigma1 * sigma2 - delta2_real * delta1_real - delta2_imag * delta1_imag

        miu_diff_real = miu2_real - miu1_real
        miu_diff_imag = miu2_imag - miu1_imag
        quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * (trace_term + quadra_term) + log_det_c2 - log_det_c1, dim=2) - self.zdim

        return kl
    
    def kl_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                        log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise):
        
        if self.latent_num == 1:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_speech, miu_noise, log_sigma_noisy_speech, log_sigma_noise, delta_noisy_speech, delta_noise, z_noisy_speech)

            kl_loss_final = self.w_kl_speech * torch.mean(kl_clean) - self.w_kl_noise * torch.mean(kl_noise) # per sample, per time frame

        elif self.latent_num == 2:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_noise, miu_noise, log_sigma_noisy_noise, log_sigma_noise, delta_noisy_noise, delta_noise, z_noisy_noise)

            kl_loss_final = self.w_kl_speech * torch.mean(kl_clean) + self.w_kl_noise *  torch.mean(kl_noise) # per sample, per time frame

        return kl_loss_final, torch.mean(kl_clean), torch.mean(kl_noise)


    def si_snr(self, source, estimate_source, eps=1e-8):
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate_source, source.t())  # B , B
        dot = torch.diagonal(dot, 0)
        dot = torch.diag(dot)
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate_source - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr) # per batch sample
        return lo    
    
    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        predict_imag = predict_cpx_stft[:,:,:,1]
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss

    def kl_loss_and_recon_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise,
                       predict_cpx_stft, ori_cpx_stft, source, est_source):
        
        kl_loss, kl_clean, kl_noise = self.kl_loss(miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise)  
        
        recon_loss, loss_cpx, loss_mag, sisnr_loss = self.multiple_recon_loss(predict_cpx_stft, ori_cpx_stft, source, est_source)

        loss = kl_loss + self.w_recon * recon_loss

        return loss, kl_loss, kl_clean, kl_noise, recon_loss, loss_cpx, loss_mag, sisnr_loss



class ete_train_se_with_latent_loss():

    def __init__(self, kl_weight, recon_loss_weight, alpha, zdim):
        self.kl_weight = kl_weight
        self.recon_loss_weight = recon_loss_weight
        self.alpha = alpha
        self.epsilon = 1e-10
        self.zdim = zdim
    
    def si_snr(self, source, estimate_source, eps=1e-8):
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate_source, source.t())  # B , B
        dot = torch.diagonal(dot, 0)
        dot = torch.diag(dot)
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate_source - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr) # per batch sample
        return lo    
    
    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        predict_imag = predict_cpx_stft[:,:,:,1]
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss
    

    def cal_kl(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

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
        abs_delta1 = torch.sqrt(delta1_real.pow(2) + delta1_imag.pow(2) + self.epsilon)
        temp = sigma1 * 0.99 / (abs_delta1 + self.epsilon)

        delta1_real = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_real * temp, delta1_real)
        delta1_imag = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_imag * temp, delta1_imag)

        abs_delta1 = delta1_real.pow(2) + delta1_imag.pow(2)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta2 = torch.sqrt(delta2_real.pow(2) + delta2_imag.pow(2) + self.epsilon)
        temp = sigma2 * 0.99 / (abs_delta2 + self.epsilon)

        delta2_real = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_real * temp, delta2_real)
        delta2_imag = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_imag * temp, delta2_imag)

        abs_delta2 = delta2_real.pow(2) + delta2_imag.pow(2)

        # B,T,H
        log_det_c1 = torch.log(0.25 * (sigma1.pow(2) - abs_delta1) + self.epsilon)
        log_det_c2 = torch.log(0.25 * (sigma2.pow(2) - abs_delta2) + self.epsilon)

        coeff = 2 / (sigma2.pow(2) - abs_delta2 + self.epsilon)


        trace_term = sigma1 * sigma2 - delta2_real * delta1_real - delta2_imag * delta1_imag

        miu_diff_real = miu2_real - miu1_real
        miu_diff_imag = miu2_imag - miu1_imag
        quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * (trace_term + quadra_term) + log_det_c2 - log_det_c1, dim=2) - self.zdim

        return kl

    def kl_loss(self, miu_clean, miu_noise, miu_noisy_speech, 
                        log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, 
                       delta_clean, delta_noise, delta_noisy_speech, 
                       z_noisy_speech):
        

        kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
        kl_noise = self.cal_kl(miu_noisy_speech, miu_noise, log_sigma_noisy_speech, log_sigma_noise, delta_noisy_speech, delta_noise, z_noisy_speech)

        kl_loss_final = torch.mean(kl_clean) - self.alpha * torch.mean(kl_noise) # per sample, per time frame


        return kl_loss_final, torch.mean(kl_clean), torch.mean(kl_noise)

    def final_ete_loss(self, miu_clean, miu_noise, miu_noisy_speech, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, 
                       delta_clean, delta_noise, delta_noisy_speech, 
                       z_noisy_speech, 
                       predict_cpx_stft, ori_cpx_stft, source, est_source):
        

        kl_loss, kl_clean, kl_noise = self.kl_loss(miu_clean, miu_noise, miu_noisy_speech, 
                         log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, 
                       delta_clean, delta_noise, delta_noisy_speech, 
                       z_noisy_speech)

        recon_loss, loss_cpx, loss_mag, sisnr_loss = self.multiple_recon_loss(predict_cpx_stft, ori_cpx_stft, source, est_source)

        final_loss = recon_loss + self.kl_weight * kl_loss
        
        return final_loss,kl_loss, kl_clean, kl_noise, recon_loss, loss_cpx, loss_mag, sisnr_loss
    



class ete_train_se_loss():

    def __init__(self, recon_loss_weight):
        self.recon_loss_weight = recon_loss_weight
        self.epsilon = 1e-10
    
    def si_snr(self, source, estimate_source, eps=1e-8):
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate_source, source.t())  # B , B
        dot = torch.diagonal(dot, 0)
        dot = torch.diag(dot)
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate_source - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr) # per batch sample
        return lo    
    
    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        predict_imag = predict_cpx_stft[:,:,:,1]
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss
    

    def final_ete_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        

        recon_loss, loss_cpx, loss_mag, sisnr_loss = self.multiple_recon_loss(predict_cpx_stft, ori_cpx_stft, source, est_source)

        
        return recon_loss, loss_cpx, loss_mag, sisnr_loss
    

class two_phase_loss():
    def __init__(self, recon_loss_weight, alpha, zdim, latent_num):
        
        self.epsilon = 1e-10
        self.recon_loss_weight = recon_loss_weight
        self.alpha = alpha
        self.zdim = zdim
        self.latent_num = latent_num

    def cal_kl(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2, z1):

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
        abs_delta1 = torch.sqrt(delta1_real.pow(2) + delta1_imag.pow(2) + self.epsilon)
        temp = sigma1 * 0.99 / (abs_delta1 + self.epsilon)

        delta1_real = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_real * temp, delta1_real)
        delta1_imag = torch.where(abs_delta1 >= (sigma1 - 1e-3), delta1_imag * temp, delta1_imag)

        abs_delta1 = delta1_real.pow(2) + delta1_imag.pow(2)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta2 = torch.sqrt(delta2_real.pow(2) + delta2_imag.pow(2) + self.epsilon)
        temp = sigma2 * 0.99 / (abs_delta2 + self.epsilon)

        delta2_real = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_real * temp, delta2_real)
        delta2_imag = torch.where(abs_delta2 >= (sigma2 - 1e-3), delta2_imag * temp, delta2_imag)

        abs_delta2 = delta2_real.pow(2) + delta2_imag.pow(2)

        # B,T,H
        log_det_c1 = torch.log(0.25 * (sigma1.pow(2) - abs_delta1) + self.epsilon)
        log_det_c2 = torch.log(0.25 * (sigma2.pow(2) - abs_delta2) + self.epsilon)

        coeff = 2 / (sigma2.pow(2) - abs_delta2 + self.epsilon)


        trace_term = sigma1 * sigma2 - delta2_real * delta1_real - delta2_imag * delta1_imag

        miu_diff_real = miu2_real - miu1_real
        miu_diff_imag = miu2_imag - miu1_imag
        quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * (trace_term + quadra_term) + log_det_c2 - log_det_c1, dim=2) - self.zdim


        return kl
    



    def si_snr(self, source, estimate_source, eps=1e-8):
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate_source, source.t())  # B , B
        dot = torch.diagonal(dot, 0)
        dot = torch.diag(dot)
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate_source - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr) # per batch sample
        return lo    
    
    def multi_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        predict_imag = predict_cpx_stft[:,:,:,1]
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss
    

    def phase_2_loss(self, predict_stft_clean, stft_x_clean, clean_batch, recon_sig_clean, predict_stft_noise, stft_x_noise, noise_batch, recon_sig_noise):
        if self.latent_num == 1:
            final_loss_clean, loss_cpx_clean, loss_mag_clean, loss_sisnr_clean = self.multi_recon_loss(predict_stft_clean, stft_x_clean, clean_batch, recon_sig_clean)
            final_loss_noise, loss_cpx_noise, loss_mag_noise, loss_sisnr_noise = torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
            final_loss = final_loss_clean
        elif self.latent_num == 2:
            final_loss_clean, loss_cpx_clean, loss_mag_clean, loss_sisnr_clean = self.multi_recon_loss(predict_stft_clean, stft_x_clean, clean_batch, recon_sig_clean)
            final_loss_noise, loss_cpx_noise, loss_mag_noise, loss_sisnr_noise = self.multi_recon_loss(predict_stft_noise, stft_x_noise, noise_batch, recon_sig_noise)
            final_loss = final_loss_clean + final_loss_noise


        return final_loss, loss_cpx_clean, loss_mag_clean, loss_sisnr_clean, loss_cpx_noise, loss_mag_noise, loss_sisnr_noise



    def phase_1_loss(self, miu_clean, miu_noise, miu_noisy_speech, miu_noisy_noise, 
                        log_sigma_clean, log_sigma_noise, log_sigma_noisy_speech, log_sigma_noisy_noise, 
                       delta_clean, delta_noise, delta_noisy_speech, delta_noisy_noise, 
                       z_noisy_speech, z_noisy_noise):
        
        if self.latent_num == 1:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_speech, miu_noise, log_sigma_noisy_speech, log_sigma_noise, delta_noisy_speech, delta_noise, z_noisy_speech)

            kl_loss_final = torch.mean(kl_clean) - self.alpha * torch.mean(kl_noise) # per sample, per time frame

        elif self.latent_num == 2:
            kl_clean = self.cal_kl(miu_noisy_speech, miu_clean, log_sigma_noisy_speech, log_sigma_clean, delta_noisy_speech, delta_clean, z_noisy_speech) #B, T
            kl_noise = self.cal_kl(miu_noisy_noise, miu_noise, log_sigma_noisy_noise, log_sigma_noise, delta_noisy_noise, delta_noise, z_noisy_noise)

            kl_loss_final = torch.mean(kl_clean) + torch.mean(kl_noise) # per sample, per time frame

        return kl_loss_final, torch.mean(kl_clean), torch.mean(kl_noise)




class adversarial_second_phase_loss():
    def __init__(self, latent_num):
        self.latent_num = latent_num

    def distinguisher_loss(self, dis_true_clean, dis_est_clean, dis_true_noise=None, dis_est_noise=None):
        loss = (dis_true_clean - 1).pow(2) + dis_est_clean.pow(2) #b,t,1
        loss = torch.mean(loss)


        return loss
    

    def si_snr(self, source, estimate_source, eps=1e-8):
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate_source, source.t())  # B , B
        dot = torch.diagonal(dot, 0)
        dot = torch.diag(dot)
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate_source - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr) # per batch sample
        return lo   

    def generator_loss(self, true_clean, est_clean, dis_est_clean, true_noise=None, est_noise=None, dis_est_noise=None):

        loss_recon = self.si_snr(true_clean, est_clean)
        loss_dis = torch.mean((dis_est_clean - 1).pow(2))
        loss = 0.5 * loss_dis + loss_recon


        return loss, loss_recon, loss_dis
