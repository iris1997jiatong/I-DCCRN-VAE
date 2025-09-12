import torch

class KL_annealing():
    def __init__(self, kl_warm_epochs):
        self.kl_warm_epochs = kl_warm_epochs
        # self.kl_warm_weights = torch.ones((kl_warm_epochs))



    def frange_cycle_linear(self, start=0.0, stop=1.0, n_cycle=1, ratio=1):

        # Linear cycling scheduler for beta (KL-divergence term annealing)

        # Credit: Fu et al., 2019: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"

        # repo: https://github.com/haofuml/cyclical_annealing

        # n_iter: number of epochs for training

        n_iter = self.kl_warm_epochs

        L = torch.ones(n_iter) * stop

        period = n_iter/n_cycle

        step = (stop-start)/(period*ratio) # linear schedule



        for c in range(n_cycle):

            v, i = start, 0

            while v <= stop and (int(i+c*period) < n_iter):

                L[int(i+c*period)] = v

                v += step

                i += 1

        return L 



 

class complex_standard_vae_loss():
    def __init__(self, kl_warm_weights, kl_weight, mi_weight, recon_loss_type='prob', recon_type='real_imag', recon_loss_weight=[1.0,1.0,1.0], num_samples=5, prior_mode='ri_inde'):
        
        self.kl_warm_weights = kl_warm_weights
        # print(kl_warm_weights.size())
        self.kl_warm_epochs = kl_warm_weights.size()[0]
        self.kl_weight = kl_weight
        self.epsilon = 1e-9
        self.recon_loss_type = recon_loss_type
        self.predict_type = recon_type
        self.recon_loss_weight = recon_loss_weight
        self.const = 1.14473
        self.num_samples = num_samples
        self.mi_weight = mi_weight
        self.prior_mode = prior_mode

    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        B, T,H,D = miu.shape
        # B*numsamples, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        # _, T, H = sigma.shape
        sigma = sigma.view(B, 1, T, H) #b*numsamples, t,h=f
        delta_real = delta[:, :, :, 0]
        delta_real = delta_real.view(B, 1, T, H)
        delta_imag = delta[:, :, :, 1]
        delta_imag = delta_imag.view(B, 1, T, H)
        z_real = z[:,:,:,:,0] # B , numsamples, T, H
        z_imag = z[:,:,:,:,1]
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_real = miu_real.view(B, 1, T, H)
        miu_imag = miu[:, :, :, 1]
        miu_imag = miu_imag.view(B, 1, T, H)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.90 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        # P = sigma
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=3)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=3)
        # log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        # log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        # miu_real = miu_real.unsqueeze(1) # B,1,T;H
        # miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        # R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        # R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        # reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part



        return log_final_prob # B; numsamples, T
    def mutual_information(self, mu, logsigma, delta, z):
        """
        Estimate I(x; z) using minibatch samples.
        Args:
            mu: [B,T,H] - encoder means
            logsigma: [B,T,H] - encoder log-variances
            delta:[B,T,H] pseudo covariance
            z_samples: [B,numsamples, T, H] - sampled latent variables
        Returns:
            mi: Scalar estimate of mutual information
        """
        
        B,T,H,D = mu.shape
        z = z.view(B, self.num_samples, T, H, D)
        # Compute log q(z|x) for each sample
        log_q_zx = self.cal_gaussian_prob(mu, logsigma, delta, z) # b,numsamples, t
        # Compute log q(z) ≈ log(mean over batch of q(z|x_j))
        log_q_z = []
        for i in range(B):
            # Compute q(z_i | x_j) for all j in the batch
            z_tmp = z[i].unsqueeze(0)
            log_prob = self.cal_gaussian_prob(mu, logsigma, delta, z_tmp) # b, numsamples, t
            log_q_z_i = torch.logsumexp(log_prob, dim=0) - torch.log(torch.tensor(B)) # numsamples, t
            log_q_z.append(log_q_z_i)
        log_q_z = torch.stack(log_q_z)  # [batch_size, numsamples, t]
        
        # MI = mean(log q(z|x) - log q(z))
        mi = torch.mean(torch.mean(log_q_zx - log_q_z, dim=1),dim=0) # t
        mi = torch.mean(mi) # per frame per sample
        return mi


    def prob_recon_loss(self, miu, input):

        if self.predict_type == 'real_imag':
            miu = torch.view_as_real(miu)
            miu_real = miu[:,:,:,0] # num_samples*B, freq, time
            miu_imag = miu[:,:,:,1]

        if self.predict_type == 'mag_wrapphase':
            pass
            # TODO

        input_real = input[:,:,:,0]
        input_imag = input[:,:,:,1]

        loss = (miu_real - input_real).pow(2) + (miu_imag - input_imag).pow(2)

        loss = torch.sum(loss, dim=1)

        loss = torch.mean(loss) # per sample per time frame

        return loss, torch.tensor(0), torch.tensor(0), torch.tensor(0)


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
    def cal_kl_arbi_prior(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2):

        # B,T,H
        self.zdim = miu1.shape[2]
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

        kl = torch.mean(kl)

        return kl    
    def complex_kl(self, miu, log_sigma, delta):
        # input shape B,T,H,2

        miu_real = miu[:,:,:,0]
        miu_imag = miu[:,:,:,1]

        sigma = torch.exp(log_sigma[:,:,:,0])

        delta_real = delta[:,:,:,0]
        delta_imag = delta[:,:,:,1]

        miu_h_miu = torch.sum(miu_real.pow(2) + miu_imag.pow(2), dim=2)

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.99 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma-1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma-1e-3), delta_imag * temp, delta_imag)

        abs_delta_square = delta_real.pow(2) + delta_imag.pow(2)        

        log_sigma_2_minus_delta_2 = torch.log(sigma.pow(2) - abs_delta_square + self.epsilon)

        res_kl = miu_h_miu + torch.sum(torch.abs(sigma - 1 - 0.5 * log_sigma_2_minus_delta_2), dim=2) # B, T

        # res_kl = torch.mean(torch.sum(res_kl, dim=1)) # per batch sample
        res_kl = torch.mean(res_kl) #per sample per time frame

        return res_kl

    def cal_loss(self, source, est_source, stft_source, miu_x, miu, log_sigma, delta, z, epoch):


        if self.recon_loss_type == 'multiple':
            recon_loss, loss_cpx, loss_mag, sisnr = self.multiple_recon_loss(miu_x, stft_source, source, est_source)
        
        if self.recon_loss_type == 'prob':
            recon_loss, loss_cpx, loss_mag, sisnr = self.prob_recon_loss(miu_x, stft_source)

        if self.prior_mode == 'ri_inde':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            # delta_prior[...,1] = 1
        elif self.prior_mode == 'ri_corr':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            delta_prior[...,1] = 1           
        kl_loss = self.cal_kl_arbi_prior(miu, miu_prior, log_sigma, log_sigma_prior, delta, delta_prior)
        # kl_loss = self.complex_kl(miu, log_sigma, delta) # lower, better (per batch sample)
        if self.mi_weight != 0:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)
        else:
            mi_loss = torch.tensor(0)

        if epoch < self.kl_warm_epochs:
            wkl = self.kl_warm_weights[epoch]
        else:
            wkl = self.kl_weight
        final_loss = recon_loss + wkl * kl_loss - self.mi_weight *  mi_loss

        

        return final_loss, recon_loss, kl_loss, mi_loss, loss_cpx, loss_mag, sisnr
    


class est_likelihood_vae_loss():
    def __init__(self, kl_warm_weights, kl_weight, mi_weight, recon_loss_type='prob', recon_type='real_imag', recon_loss_weight=[1.0,1.0,1.0], num_samples=5, prior_mode='ri_inde'):
        
        self.kl_warm_weights = kl_warm_weights
        # print(kl_warm_weights.size())
        self.kl_warm_epochs = kl_warm_weights.size()[0]
        self.kl_weight = kl_weight
        self.epsilon = 1e-10
        self.recon_loss_type = recon_loss_type
        self.predict_type = recon_type
        self.recon_loss_weight = recon_loss_weight
        self.const = 1.14473
        self.num_samples = num_samples
        self.mi_weight = mi_weight
        self.prior_mode = prior_mode   
    def mutual_information(self, mu, logsigma, delta, z):
        """
        Estimate I(x; z) using minibatch samples.
        Args:
            mu: [B,T,H] - encoder means
            logsigma: [B,T,H] - encoder log-variances
            delta:[B,T,H] pseudo covariance
            z_samples: [B,numsamples, T, H] - sampled latent variables
        Returns:
            mi: Scalar estimate of mutual information
        """
        
        B,T,H,D = mu.shape
        z = z.view(B, self.num_samples, T, H, D)
        # Compute log q(z|x) for each sample
        log_q_zx = self.cal_gaussian_prob(mu, logsigma, delta, z) # b,numsamples, t
        # Compute log q(z) ≈ log(mean over batch of q(z|x_j))
        log_q_z = []
        for i in range(B):
            # Compute q(z_i | x_j) for all j in the batch
            z_tmp = z[i].unsqueeze(0)
            log_prob = self.cal_gaussian_prob(mu, logsigma, delta, z_tmp) # b, numsamples, t
            log_q_z_i = torch.logsumexp(log_prob, dim=0) - torch.log(torch.tensor(B)) # numsamples, t
            log_q_z.append(log_q_z_i)
        log_q_z = torch.stack(log_q_z)  # [batch_size, numsamples, t]
        
        # MI = mean(log q(z|x) - log q(z))
        mi = torch.mean(torch.mean(log_q_zx - log_q_z, dim=1),dim=0) # t
        mi = torch.mean(mi) # per frame per sample
        return mi
    def cal_kl_arbi_prior(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2):

        # B,T,H
        self.zdim = miu1.shape[2]
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

        kl = torch.mean(kl)

        return kl    
    def check_and_log_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            # print(f"Tensor contents:\n{tensor}")
            # self.detect_anormal = False
            raise RuntimeError(f"NaN detected in {name}")
    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        z = z.permute(0,1,3,2,4) #b,numsamples, t,f,2
        B = z.shape[0]
        # B*numsamples, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        _, T, H = sigma.shape
        sigma = sigma.view(B, self.num_samples, T, H) #b*numsamples, t,h=f
        delta_real = delta[:, :, :, 0]
        delta_real = delta_real.view(B, self.num_samples, T, H)
        delta_imag = delta[:, :, :, 1]
        delta_imag = delta_imag.view(B, self.num_samples, T, H)
        z_real = z[:,:,:,:,0] # B , numsamples, T, H
        z_imag = z[:,:,:,:,1]
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_real = miu_real.view(B, self.num_samples, T, H)
        miu_imag = miu[:, :, :, 1]
        miu_imag = miu_imag.view(B, self.num_samples, T, H)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.90 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        # P = sigma
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=3)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=3)
        # log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        # log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        # miu_real = miu_real.unsqueeze(1) # B,1,T;H
        # miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        # R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        # R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        # reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part



   
        loss_cpx = torch.sum(z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2), dim=3)


        return log_final_prob, loss_cpx # B; numsamples, T
    def prob_recon_loss(self, miu_x, log_sigma_x, delta_x, input_x):

        log_prob, loss_cpx = self.cal_gaussian_prob(miu_x, log_sigma_x, delta_x, input_x)
        prob_mean = torch.mean(log_prob)
        loss_cpx = torch.mean(loss_cpx)
        return -prob_mean, loss_cpx, torch.tensor(0), torch.tensor(0)

    def cal_loss(self, source, est_source, stft_source, miu_x, log_sigma_x, delta_x, miu, log_sigma, delta, z, epoch):
        if self.recon_loss_type == 'multiple':
            pass
            # recon_loss, loss_cpx, loss_mag, sisnr = self.multiple_recon_loss(miu_x, stft_source, source, est_source)
        
        if self.recon_loss_type == 'prob':
            recon_loss, loss_cpx, loss_mag, sisnr = self.prob_recon_loss(miu_x, log_sigma_x, delta_x, stft_source)

        if self.prior_mode == 'ri_inde':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            # delta_prior[...,1] = 1
        elif self.prior_mode == 'ri_corr':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            delta_prior[...,1] = 1           
        kl_loss = self.cal_kl_arbi_prior(miu, miu_prior, log_sigma, log_sigma_prior, delta, delta_prior)
        # kl_loss = self.complex_kl(miu, log_sigma, delta) # lower, better (per batch sample)
        if self.mi_weight != 0:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)
        else:
            mi_loss = torch.tensor(0)

        if epoch < self.kl_warm_epochs:
            wkl = self.kl_warm_weights[epoch]
        else:
            wkl = self.kl_weight
        final_loss = recon_loss + wkl * kl_loss - self.mi_weight *  mi_loss

        return final_loss, recon_loss, kl_loss, mi_loss, loss_cpx, loss_mag, sisnr
    



class  complex_vcae_loss():
    def __init__(self, kl_weight, mi_weight, recon_loss_type='multiple', recon_type='real_imag', recon_loss_weight=[1.0,1.0,0.0], num_samples=5, prior_mode='ri_inde', pz_sigma=1):

        self.kl_weight = kl_weight
        self.epsilon = 1e-9
        self.recon_loss_type = recon_loss_type
        self.predict_type = recon_type
        self.recon_loss_weight = recon_loss_weight
        self.const = 1.14473
        self.num_samples = num_samples
        self.mi_weight = mi_weight
        self.prior_mode = prior_mode
        self.pz_sigma = pz_sigma

    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        # z = z.permute(0,1,3,2,4) #b,numsamples, t,f,2
        B, T,H,D = miu.shape
        # z = z.view(B, self.num_samples,T, H, D)
        # B*numsamples, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        sigma = sigma.view(B, 1, T, H) #b*numsamples, t,h=f
        delta_real = delta[:, :, :, 0]
        delta_real = delta_real.view(B, 1, T, H)
        delta_imag = delta[:, :, :, 1]
        delta_imag = delta_imag.view(B, 1, T, H)
        z_real = z[:,:,:,:,0] # B , numsamples, T, H
        z_imag = z[:,:,:,:,1]
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_real = miu_real.view(B, 1, T, H)
        miu_imag = miu[:, :, :, 1]
        miu_imag = miu_imag.view(B, 1, T, H)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.90 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        # P = sigma
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=3)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=3)
        # log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        # log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        # miu_real = miu_real.unsqueeze(1) # B,1,T;H
        # miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        # R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        # R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        # reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part



        return log_final_prob # B; numsamples, T
    
    def mutual_information(self, mu, logsigma, delta, z):
        """
        Estimate I(x; z) using minibatch samples.
        Args:
            mu: [B,T,H] - encoder means
            logsigma: [B,T,H] - encoder log-variances
            delta:[B,T,H] pseudo covariance
            z_samples: [B,numsamples, T, H] - sampled latent variables
        Returns:
            mi: Scalar estimate of mutual information
        """
        
        B,T,H,D = mu.shape
        z = z.view(B, self.num_samples, T, H, D)
        # Compute log q(z|x) for each sample
        log_q_zx = self.cal_gaussian_prob(mu, logsigma, delta, z) # b,numsamples, t
        # Compute log q(z) ≈ log(mean over batch of q(z|x_j))
        log_q_z = []
        for i in range(B):
            # Compute q(z_i | x_j) for all j in the batch
            z_tmp = z[i].unsqueeze(0)
            log_prob = self.cal_gaussian_prob(mu, logsigma, delta, z_tmp) # b, numsamples, t
            log_q_z_i = torch.logsumexp(log_prob, dim=0) - torch.log(torch.tensor(B)) # numsamples, t
            log_q_z.append(log_q_z_i)
        log_q_z = torch.stack(log_q_z)  # [batch_size, numsamples, t]
        
        # MI = mean(log q(z|x) - log q(z))
        mi = torch.mean(torch.mean(log_q_zx - log_q_z, dim=1),dim=0) # t
        mi = torch.mean(mi) # per frame per sample
        return mi


    def prob_recon_loss(self, miu, input):

        if self.predict_type == 'real_imag':
            miu = torch.view_as_real(miu)
            miu_real = miu[:,:,:,0] # num_samples*B, freq, time
            miu_imag = miu[:,:,:,1]

        if self.predict_type == 'mag_wrapphase':
            pass
            # TODO

        input_real = input[:,:,:,0]
        input_imag = input[:,:,:,1]

        loss = (miu_real - input_real).pow(2) + (miu_imag - input_imag).pow(2)
        # loss = torch.abs(miu_real - input_real) + torch.abs(miu_real - input_real)

        loss = torch.sum(loss, dim=1)

        loss = torch.mean(loss) # per sample per time frame

        return loss, torch.tensor(0), torch.tensor(0), torch.tensor(0)


    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        # predict_real = torch.sign(predict_real) * torch.abs(predict_real + self.epsilon) ** (0.4)
        predict_imag = predict_cpx_stft[:,:,:,1]
        # predict_imag = torch.sign(predict_imag) * torch.abs(predict_imag + self.epsilon) ** (0.4)
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)
        # predict_mag = torch.log10(predict_real.pow(2) + predict_imag.pow(2) + self.epsilon)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        # ori_stft_real = torch.sign(ori_stft_real) * torch.abs(ori_stft_real) ** (0.4)
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        # ori_stft_imag = torch.sign(ori_stft_imag) * torch.abs(ori_stft_imag) ** (0.4)
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)
        # ori_mag = torch.log10(ori_stft_real.pow(2) + ori_stft_real.pow(2) + self.epsilon)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum((torch.sign(predict_real)*torch.log10(torch.abs(predict_real)+self.epsilon) - torch.sign(ori_stft_real)*torch.log10(torch.abs(ori_stft_real)+self.epsilon)).pow(2),dim=1) + torch.sum((torch.sign(predict_imag)*torch.log10(torch.abs(predict_imag)+self.epsilon) - torch.sign(ori_stft_imag)*torch.log10(torch.abs(ori_stft_imag)+self.epsilon)).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss





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
    def cal_kl_arbi_prior(self, log_sigma1, log_sigma2, delta1, delta2):

        # B,T,H
        self.zdim = log_sigma1.shape[2]
        # miu1_real = miu1[:,:,:,0]
        # miu1_imag = miu1[:,:,:,1]

        # miu2_real = miu2[:,:,:,0]
        # miu2_imag = miu2[:,:,:,1]

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

        # miu_diff_real = miu2_real - miu1_real
        # miu_diff_imag = miu2_imag - miu1_imag
        # quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * trace_term + log_det_c2 - log_det_c1, dim=2) - self.zdim

        kl = torch.mean(kl)

        return kl    
    def complex_kl(self, miu, log_sigma, delta):
        # input shape B,T,H,2

        miu_real = miu[:,:,:,0]
        miu_imag = miu[:,:,:,1]

        sigma = torch.exp(log_sigma[:,:,:,0])

        delta_real = delta[:,:,:,0]
        delta_imag = delta[:,:,:,1]

        miu_h_miu = torch.sum(miu_real.pow(2) + miu_imag.pow(2), dim=2)

        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.99 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma-1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma-1e-3), delta_imag * temp, delta_imag)

        abs_delta_square = delta_real.pow(2) + delta_imag.pow(2)        

        log_sigma_2_minus_delta_2 = torch.log(sigma.pow(2) - abs_delta_square + self.epsilon)

        res_kl = miu_h_miu + torch.sum(torch.abs(sigma - 1 - 0.5 * log_sigma_2_minus_delta_2), dim=2) # B, T

        # res_kl = torch.mean(torch.sum(res_kl, dim=1)) # per batch sample
        res_kl = torch.mean(res_kl) #per sample per time frame

        return res_kl

    def cal_loss(self, source, est_source, stft_source, miu_x, miu, log_sigma, delta, z):


        if self.recon_loss_type == 'multiple':
            recon_loss, loss_cpx, loss_mag, sisnr = self.multiple_recon_loss(miu_x, stft_source, source, est_source)
        
        if self.recon_loss_type == 'prob':
            recon_loss, loss_cpx, loss_mag, sisnr = self.prob_recon_loss(miu_x, stft_source)

        if self.prior_mode == 'ri_inde':
            # miu_prior = torch.zeros_like(miu)
            log_sigma_define = torch.log(torch.tensor(self.pz_sigma))
            log_sigma_prior = torch.ones_like(log_sigma) * log_sigma_define
            delta_prior = torch.zeros_like(delta)
            # delta_prior[...,1] = 1
        elif self.prior_mode == 'ri_corr':
            # miu_prior = torch.zeros_like(miu)
            log_sigma_define = torch.log(self.pz_sigma)
            log_sigma_prior = torch.ones_like(log_sigma) * log_sigma_define
            delta_prior = torch.zeros_like(delta)
            delta_prior[...,1] = log_sigma_define           
        kl_loss = self.cal_kl_arbi_prior(log_sigma, log_sigma_prior, delta, delta_prior)
        # kl_loss = self.complex_kl(miu, log_sigma, delta) # lower, better (per batch sample)
        if self.mi_weight != 0:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)
        else:
            mi_loss = torch.tensor(0)


        wkl = self.kl_weight
        final_loss = recon_loss + wkl * kl_loss - self.mi_weight *  mi_loss

        

        return final_loss, recon_loss, kl_loss, mi_loss, loss_cpx, loss_mag, sisnr
    



class  complex_vcae_regmiu_loss():
    def __init__(self, kl_weight, mi_weight, regmiu_w, recon_loss_type='multiple', recon_type='real_imag', recon_loss_weight=[1.0,1.0,0.0], num_samples=5, prior_mode='ri_inde', pz_sigma=1, miu_sigma=1, loss_opt=1):

        self.kl_weight = kl_weight
        self.epsilon = 1e-9
        self.recon_loss_type = recon_loss_type
        self.predict_type = recon_type
        self.recon_loss_weight = recon_loss_weight
        self.const = 1.14473
        self.num_samples = num_samples
        self.mi_weight = mi_weight
        self.prior_mode = prior_mode
        self.pz_sigma = pz_sigma
        self.miu_sigma = miu_sigma
        self.loss_opt = loss_opt
        self.regmiu_w = regmiu_w

    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        # z = z.permute(0,1,3,2,4) #b,numsamples, t,f,2
        B, T,H,D = miu.shape
        # z = z.view(B, self.num_samples,T, H, D)
        # B*numsamples, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        sigma = sigma.view(B, 1, T, H) #b*numsamples, t,h=f
        delta_real = delta[:, :, :, 0]
        delta_real = delta_real.view(B, 1, T, H)
        delta_imag = delta[:, :, :, 1]
        delta_imag = delta_imag.view(B, 1, T, H)
        z_real = z[:,:,:,:,0] # B , numsamples, T, H
        z_imag = z[:,:,:,:,1]
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_real = miu_real.view(B, 1, T, H)
        miu_imag = miu[:, :, :, 1]
        miu_imag = miu_imag.view(B, 1, T, H)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.90 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        # P = sigma
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=3)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=3)
        # log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        # log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        # miu_real = miu_real.unsqueeze(1) # B,1,T;H
        # miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        # R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        # R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        # reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part



        return log_final_prob # B; numsamples, T
    
    def mutual_information(self, mu, logsigma, delta, z):
        """
        Estimate I(x; z) using minibatch samples.
        Args:
            mu: [B,T,H] - encoder means
            logsigma: [B,T,H] - encoder log-variances
            delta:[B,T,H] pseudo covariance
            z_samples: [B,numsamples, T, H] - sampled latent variables
        Returns:
            mi: Scalar estimate of mutual information
        """
        
        B,T,H,D = mu.shape
        z = z.view(B, self.num_samples, T, H, D)
        # Compute log q(z|x) for each sample
        log_q_zx = self.cal_gaussian_prob(mu, logsigma, delta, z) # b,numsamples, t
        # Compute log q(z) ≈ log(mean over batch of q(z|x_j))
        log_q_z = []
        for i in range(B):
            # Compute q(z_i | x_j) for all j in the batch
            z_tmp = z[i].unsqueeze(0)
            log_prob = self.cal_gaussian_prob(mu, logsigma, delta, z_tmp) # b, numsamples, t
            log_q_z_i = torch.logsumexp(log_prob, dim=0) - torch.log(torch.tensor(B)) # numsamples, t
            log_q_z.append(log_q_z_i)
        log_q_z = torch.stack(log_q_z)  # [batch_size, numsamples, t]
        
        # MI = mean(log q(z|x) - log q(z))
        mi = torch.mean(torch.mean(log_q_zx - log_q_z, dim=1),dim=0) # t
        mi = torch.mean(mi) # per frame per sample
        return mi


    def prob_recon_loss(self, miu, input):

        if self.predict_type == 'real_imag':
            miu = torch.view_as_real(miu)
            miu_real = miu[:,:,:,0] # num_samples*B, freq, time
            miu_imag = miu[:,:,:,1]

        if self.predict_type == 'mag_wrapphase':
            pass
            # TODO

        input_real = input[:,:,:,0]
        input_imag = input[:,:,:,1]

        loss = (miu_real - input_real).pow(2) + (miu_imag - input_imag).pow(2)
        # loss = torch.abs(miu_real - input_real) + torch.abs(miu_real - input_real)

        loss = torch.sum(loss, dim=1)

        loss = torch.mean(loss) # per sample per time frame

        return loss, torch.tensor(0), torch.tensor(0), torch.tensor(0)


    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        # predict_real = torch.sign(predict_real) * torch.abs(predict_real + self.epsilon) ** (0.4)
        predict_imag = predict_cpx_stft[:,:,:,1]
        # predict_imag = torch.sign(predict_imag) * torch.abs(predict_imag + self.epsilon) ** (0.4)
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)
        # predict_mag = torch.log10(predict_real.pow(2) + predict_imag.pow(2) + self.epsilon)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        # ori_stft_real = torch.sign(ori_stft_real) * torch.abs(ori_stft_real) ** (0.4)
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        # ori_stft_imag = torch.sign(ori_stft_imag) * torch.abs(ori_stft_imag) ** (0.4)
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)
        # ori_mag = torch.log10(ori_stft_real.pow(2) + ori_stft_real.pow(2) + self.epsilon)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum((torch.sign(predict_real)*torch.log10(torch.abs(predict_real)+self.epsilon) - torch.sign(ori_stft_real)*torch.log10(torch.abs(ori_stft_real)+self.epsilon)).pow(2),dim=1) + torch.sum((torch.sign(predict_imag)*torch.log10(torch.abs(predict_imag)+self.epsilon) - torch.sign(ori_stft_imag)*torch.log10(torch.abs(ori_stft_imag)+self.epsilon)).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss





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
    def cal_kl_arbi_prior(self, log_sigma1, log_sigma2, delta1, delta2):

        # B,T,H
        self.zdim = log_sigma1.shape[2]
        # miu1_real = miu1[:,:,:,0]
        # miu1_imag = miu1[:,:,:,1]

        # miu2_real = miu2[:,:,:,0]
        # miu2_imag = miu2[:,:,:,1]

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

        # miu_diff_real = miu2_real - miu1_real
        # miu_diff_imag = miu2_imag - miu1_imag
        # quadra_term = miu_diff_real.pow(2) * (sigma2 - delta2_real) - 2 * delta2_imag * miu_diff_real * miu_diff_imag + miu_diff_imag.pow(2)*(sigma2 + delta2_real)

        kl = 0.5 * torch.sum(coeff * trace_term + log_det_c2 - log_det_c1, dim=2) - self.zdim

        kl = torch.mean(kl)

        return kl    

    def miu_regu_loss(self, miu):
        # miu [b, t, h, 2]
        B, T, H, D = miu.shape
        num_latents = B*T
        miu = miu.reshape(num_latents, H, D)
        mean_miu = torch.mean(miu, dim=0, keepdim=True) # 1, H, 2
        miu_minus_meanmiu = miu - mean_miu
        real_diff = miu_minus_meanmiu[..., 0] # b*t, H
        imag_diff = miu_minus_meanmiu[..., 1]
        diff_vector = torch.cat([real_diff, imag_diff], dim=1) # b*t, 2H
        cov_miu = torch.matmul(diff_vector.T, diff_vector) / num_latents #2h * 2h
        diag_elements = torch.diagonal(cov_miu, 0)
        offdiag = cov_miu - torch.diag(diag_elements)

        if self.loss_opt == 1:
            avg_diag = torch.mean(diag_elements)
            diag_loss = (avg_diag - self.miu_sigma).pow(2)
            avg_offdiag = torch.mean(offdiag)
            offdiag_loss = avg_offdiag.pow(2)

        elif self.loss_opt == 2:
            diag_loss = torch.mean((diag_elements - self.miu_sigma).pow(2))
            offdiag_loss = torch.mean(offdiag.pow(2))

        reguloss = 0 * offdiag_loss + self.regmiu_w * diag_loss

        return reguloss, offdiag_loss, diag_loss
      


    def cal_loss(self, source, est_source, stft_source, miu_x, miu, log_sigma, delta, z):


        if self.recon_loss_type == 'multiple':
            recon_loss, loss_cpx, loss_mag, sisnr = self.multiple_recon_loss(miu_x, stft_source, source, est_source)
        
        if self.recon_loss_type == 'prob':
            recon_loss, loss_cpx, loss_mag, sisnr = self.prob_recon_loss(miu_x, stft_source)

        if self.prior_mode == 'ri_inde':
            # miu_prior = torch.zeros_like(miu)
            log_sigma_define = torch.log(torch.tensor(self.pz_sigma))
            log_sigma_prior = torch.ones_like(log_sigma) * log_sigma_define
            delta_prior = torch.zeros_like(delta)
            # delta_prior[...,1] = 1
        elif self.prior_mode == 'ri_corr':
            # miu_prior = torch.zeros_like(miu)
            log_sigma_define = torch.log(self.pz_sigma)
            log_sigma_prior = torch.ones_like(log_sigma) * log_sigma_define
            delta_prior = torch.zeros_like(delta)
            delta_prior[...,1] = log_sigma_define           
        kl_loss = self.cal_kl_arbi_prior(log_sigma, log_sigma_prior, delta, delta_prior)
        # kl_loss = self.complex_kl(miu, log_sigma, delta) # lower, better (per batch sample)
        if self.mi_weight != 0:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)
        else:
            mi_loss = torch.tensor(0)

        miu_loss, offloss, diagloss = self.miu_regu_loss(miu)
        wkl = self.kl_weight
        final_loss = recon_loss + wkl * kl_loss + miu_loss - self.mi_weight *  mi_loss

        

        return final_loss, recon_loss, kl_loss, offloss, diagloss, mi_loss, loss_cpx, loss_mag, sisnr
    


class  complex_dip_vae_loss():
    def __init__(self, kl_weight, mi_weight, off_weight, diag_weight, recon_loss_type='multiple', recon_type='real_imag', 
                 recon_loss_weight=[1.0,1.0,0.0], num_samples=5, prior_mode='ri_inde', miu_sigma=1):

        self.kl_weight = kl_weight
        self.epsilon = 1e-9
        self.recon_loss_type = recon_loss_type
        self.predict_type = recon_type
        self.recon_loss_weight = recon_loss_weight
        self.const = 1.14473
        self.num_samples = num_samples
        self.mi_weight = mi_weight
        self.prior_mode = prior_mode
        self.miu_sigma = miu_sigma
        self.off_weight = off_weight
        self.diag_weight = diag_weight
        self.off_weight = off_weight

    def cal_gaussian_prob(self, miu, log_sigma, delta, z):

        # z = z.permute(0,1,3,2,4) #b,numsamples, t,f,2
        B, T,H,D = miu.shape
        # z = z.view(B, self.num_samples,T, H, D)
        # B*numsamples, T, H
        sigma = torch.exp(log_sigma[:, :, :, 0])
        sigma = sigma.view(B, 1, T, H) #b*numsamples, t,h=f
        delta_real = delta[:, :, :, 0]
        delta_real = delta_real.view(B, 1, T, H)
        delta_imag = delta[:, :, :, 1]
        delta_imag = delta_imag.view(B, 1, T, H)
        z_real = z[:,:,:,:,0] # B , numsamples, T, H
        z_imag = z[:,:,:,:,1]
        miu_real = miu[:, :, :, 0] # B;T;H
        miu_real = miu_real.view(B, 1, T, H)
        miu_imag = miu[:, :, :, 1]
        miu_imag = miu_imag.view(B, 1, T, H)

        # protection
        # keep abs(delta) <= sigma (protection)
        abs_delta = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.epsilon)
        temp = sigma * 0.90 / (abs_delta + self.epsilon)

        delta_real = torch.where(abs_delta >= (sigma - 1e-3), delta_real * temp, delta_real)
        delta_imag = torch.where(abs_delta >= (sigma - 1e-3), delta_imag * temp, delta_imag)

        abs_delta = delta_real.pow(2) + delta_imag.pow(2)
        P = sigma - abs_delta / (sigma + self.epsilon)
        # P = sigma
        reci_p = 1 / (P + self.epsilon)
        R_P_minus_1_real = delta_real / (sigma * P + self.epsilon) # B,T;H
        R_P_minus_1_imag = ((-1) * delta_imag) / (sigma * P + self.epsilon)
        p_1_minus_RPR = reci_p - abs_delta / (sigma * P * sigma + self.epsilon)
        # det_p_1_minus_RPR = torch.prod(p_1_minus_RPR, dim=2, keepdim=False) # B, T
        log_det_p_1_minus_RPR = torch.sum(torch.log(p_1_minus_RPR + self.epsilon), dim=3)
        # det_1_over_p = torch.prod(reci_p, dim=2, keepdim=False) # B, T
        log_1_over_p = torch.sum(torch.log(reci_p + self.epsilon), dim=3)
        # log_det_p_1_minus_RPR = log_det_p_1_minus_RPR.unsqueeze(1) # B,1,T
        # log_1_over_p = log_1_over_p.unsqueeze(1) # B,1,T

        # miu_real = miu_real.unsqueeze(1) # B,1,T;H
        # miu_imag = miu_imag.unsqueeze(1) # B,1,T;H

        z_minus_miu_real = z_real - miu_real # B,numsamples,T;H
        z_minus_miu_imag = z_imag - miu_imag

        # R_P_minus_1_real = R_P_minus_1_real.unsqueeze(1) # B,1,T;H
        # R_P_minus_1_imag = R_P_minus_1_imag.unsqueeze(1) # B,1,T;H

        # reci_p = reci_p.unsqueeze(1) # B,1,T;H

        z_minus_miu_P_z_minus_miu = torch.sum((z_minus_miu_real.pow(2) + z_minus_miu_imag.pow(2)) * reci_p, dim=3) * (-1) # B, numsamples, T
        real_exp_part = torch.sum((z_minus_miu_real.pow(2) - z_minus_miu_imag.pow(2)) * R_P_minus_1_real - 2 * z_minus_miu_real * z_minus_miu_imag * R_P_minus_1_imag, dim=3) # B, num_samples, T;H --> B, numsamples, T
        real_exp_part = real_exp_part + z_minus_miu_P_z_minus_miu # B, numsamples, T

        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon)/ self.pi_n * torch.exp(real_exp_part) # B,numsamples, T
        # final_prob = torch.sqrt(det_p_1_minus_RPR * det_1_over_p + self.epsilon) * torch.exp(real_exp_part) # B,numsamples, T

        
        # log_final_prob = 0.5 * torch.log(det_p_1_minus_RPR * det_1_over_p + self.epsilon) + real_exp_part
        log_final_prob = 0.5 * (log_det_p_1_minus_RPR + log_1_over_p) + real_exp_part



        return log_final_prob # B; numsamples, T
    
    def mutual_information(self, mu, logsigma, delta, z):
        """
        Estimate I(x; z) using minibatch samples.
        Args:
            mu: [B,T,H] - encoder means
            logsigma: [B,T,H] - encoder log-variances
            delta:[B,T,H] pseudo covariance
            z_samples: [B,numsamples, T, H] - sampled latent variables
        Returns:
            mi: Scalar estimate of mutual information
        """
        
        B,T,H,D = mu.shape
        z = z.view(B, self.num_samples, T, H, D)
        # Compute log q(z|x) for each sample
        log_q_zx = self.cal_gaussian_prob(mu, logsigma, delta, z) # b,numsamples, t
        # Compute log q(z) ≈ log(mean over batch of q(z|x_j))
        log_q_z = []
        for i in range(B):
            # Compute q(z_i | x_j) for all j in the batch
            z_tmp = z[i].unsqueeze(0)
            log_prob = self.cal_gaussian_prob(mu, logsigma, delta, z_tmp) # b, numsamples, t
            log_q_z_i = torch.logsumexp(log_prob, dim=0) - torch.log(torch.tensor(B)) # numsamples, t
            log_q_z.append(log_q_z_i)
        log_q_z = torch.stack(log_q_z)  # [batch_size, numsamples, t]
        
        # MI = mean(log q(z|x) - log q(z))
        mi = torch.mean(torch.mean(log_q_zx - log_q_z, dim=1),dim=0) # t
        mi = torch.mean(mi) # per frame per sample
        return mi


    def prob_recon_loss(self, miu, input):

        if self.predict_type == 'real_imag':
            miu = torch.view_as_real(miu)
            miu_real = miu[:,:,:,0] # num_samples*B, freq, time
            miu_imag = miu[:,:,:,1]

        if self.predict_type == 'mag_wrapphase':
            pass
            # TODO

        input_real = input[:,:,:,0]
        input_imag = input[:,:,:,1]

        loss = (miu_real - input_real).pow(2) + (miu_imag - input_imag).pow(2)
        # loss = torch.abs(miu_real - input_real) + torch.abs(miu_real - input_real)

        loss = torch.sum(loss, dim=1)

        loss = torch.mean(loss) # per sample per time frame

        return loss, torch.tensor(0), torch.tensor(0), torch.tensor(0)


    def multiple_recon_loss(self, predict_cpx_stft, ori_cpx_stft, source, est_source):
        predict_cpx_stft = torch.view_as_real(predict_cpx_stft)
        predict_real = predict_cpx_stft[:,:,:,0]
        # predict_real = torch.sign(predict_real) * torch.abs(predict_real + self.epsilon) ** (0.4)
        predict_imag = predict_cpx_stft[:,:,:,1]
        # predict_imag = torch.sign(predict_imag) * torch.abs(predict_imag + self.epsilon) ** (0.4)
        predict_mag = torch.sqrt(predict_real.pow(2) + predict_imag.pow(2) + 1e-6)
        # predict_mag = torch.log10(predict_real.pow(2) + predict_imag.pow(2) + self.epsilon)

        ori_stft_real = ori_cpx_stft[:,:,:,0]
        # ori_stft_real = torch.sign(ori_stft_real) * torch.abs(ori_stft_real) ** (0.4)
        ori_stft_imag = ori_cpx_stft[:,:,:,1]
        # ori_stft_imag = torch.sign(ori_stft_imag) * torch.abs(ori_stft_imag) ** (0.4)
        ori_mag = torch.sqrt(ori_stft_real.pow(2) + ori_stft_real.pow(2) + 1e-6)
        # ori_mag = torch.log10(ori_stft_real.pow(2) + ori_stft_real.pow(2) + self.epsilon)

        loss_cpx = torch.sum((predict_real - ori_stft_real).pow(2),dim=1) + torch.sum((predict_imag - ori_stft_imag).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum((torch.sign(predict_real)*torch.log10(torch.abs(predict_real)+self.epsilon) - torch.sign(ori_stft_real)*torch.log10(torch.abs(ori_stft_real)+self.epsilon)).pow(2),dim=1) + torch.sum((torch.sign(predict_imag)*torch.log10(torch.abs(predict_imag)+self.epsilon) - torch.sign(ori_stft_imag)*torch.log10(torch.abs(ori_stft_imag)+self.epsilon)).pow(2),dim=1) # batch x time
        # loss_cpx = torch.sum(torch.abs(predict_real - ori_stft_real),dim=1) + torch.sum(torch.abs(predict_imag - ori_stft_imag),dim=1) # batch x time
        loss_cpx = torch.mean(loss_cpx) # per sample and per time frame

        loss_mag = torch.sum((predict_mag - ori_mag).pow(2), dim=1) # batch, time
        # loss_mag = torch.sum(torch.abs(predict_mag - ori_mag), dim=1) # batch, time
        loss_mag = torch.mean(loss_mag) # per sample, per time frame

        sisnr_loss = self.si_snr(source, est_source) # per sample

        final_loss = self.recon_loss_weight[0] * loss_cpx + self.recon_loss_weight[1] * loss_mag + self.recon_loss_weight[2] * sisnr_loss

        return final_loss, loss_cpx, loss_mag, sisnr_loss





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
    def cal_kl_arbi_prior(self, miu1, miu2, log_sigma1, log_sigma2, delta1, delta2):

        # B,T,H
        self.zdim = miu1.shape[2]
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

        kl = torch.mean(kl)

        return kl    

    def miu_regu_loss(self, miu):
        # miu [b, t, h, 2]
        B, T, H, D = miu.shape
        num_latents = B*T
        miu = miu.reshape(num_latents, H, D)
        mean_miu = torch.mean(miu, dim=0, keepdim=True) # 1, H, 2
        miu_minus_meanmiu = miu - mean_miu
        real_diff = miu_minus_meanmiu[..., 0] # b*t, H
        imag_diff = miu_minus_meanmiu[..., 1]
        diff_vector = torch.cat([real_diff, imag_diff], dim=1) # b*t, 2H
        cov_miu = torch.matmul(diff_vector.T, diff_vector) / num_latents #2h * 2h
        diag_elements = torch.diagonal(cov_miu, 0)
        offdiag = cov_miu - torch.diag(diag_elements)

        # LOSS 4 no clamp and sqrt
        # avg_diag = torch.mean(diag_elements)
        # diag_loss = (avg_diag - self.miu_sigma).pow(2)
        # diag_loss = torch.clamp(diag_loss, min=1e-8)
        # diag_loss = torch.sqrt(diag_loss)

        # avg_offdiag = torch.mean(offdiag)
        # offdiag_loss = avg_offdiag.pow(2)
        # offdiag_loss = torch.clamp(offdiag_loss, min=1e-8)
        # offdiag_loss = torch.sqrt(offdiag_loss)

        # reguloss = self.off_weight * offdiag_loss + self.diag_weight * diag_loss
        # loss 2
        # diag_loss = torch.clamp((diag_elements - self.miu_sigma).pow(2), min=1e-8)
        # diag_loss = torch.mean(diag_loss)
        # diag_loss = torch.sqrt(diag_loss)
        # offdiag_loss = torch.clamp(offdiag.pow(2), min=1e-8)
        # offdiag_loss = torch.mean(offdiag_loss)
        # offdiag_loss = torch.sqrt(offdiag_loss) 
        #            
        # loss 3
        # diag_loss = (diag_elements - self.miu_sigma).pow(2)
        # diag_loss = torch.mean(diag_loss)
        # # diag_loss = torch.sqrt(diag_loss)
        # offdiag_loss = offdiag.pow(2)
        # offdiag_loss = torch.mean(offdiag_loss)
        # # offdiag_loss = torch.sqrt(offdiag_loss)   
        # 
        # loss 5
        # avg_diag = torch.mean(diag_elements)
        # diag_loss = (avg_diag - self.miu_sigma).pow(2)  
        # diag_loss = torch.clamp(diag_loss, min=1e-8)
        # diag_loss = torch.sqrt(diag_loss)   
        # offdiag_loss = torch.clamp(offdiag.pow(2), min=1e-8)
        # offdiag_loss = torch.mean(offdiag_loss)
        # offdiag_loss = torch.sqrt(offdiag_loss)  

        # loss 6
        avg_diag = torch.mean(diag_elements)
        diag_loss = (avg_diag - self.miu_sigma).pow(2)  
        diag_loss = torch.clamp(diag_loss, min=1e-8)
        diag_loss = torch.sqrt(diag_loss)  
        vri = offdiag[:H, H:]
        avg_vri_abs_diag = torch.mean(torch.abs(torch.diagonal(vri,0)))
        vri_diag_loss = (avg_vri_abs_diag - self.miu_sigma).pow(2)
        vri_diag_loss = torch.clamp(vri_diag_loss, min=1e-8)
        vri_diag_loss = torch.sqrt(vri_diag_loss)   
        diag_loss = (diag_loss + vri_diag_loss) * 0.5

        offdiag = offdiag.clone()
        offdiag[:H, H:] = offdiag[:H, H:] - torch.diag(torch.diagonal(offdiag[:H, H:], 0))
        offdiag[H:, :H] = offdiag[H:, :H] - torch.diag(torch.diagonal(offdiag[H:, :H], 0))

        offdiag_loss = torch.clamp(offdiag.pow(2), min=1e-8)
        offdiag_loss = torch.mean(offdiag_loss) 
        offdiag_loss = torch.sqrt(offdiag_loss)          



        reguloss = self.off_weight * offdiag_loss + self.diag_weight * diag_loss

        return reguloss, offdiag_loss, diag_loss
      


    def cal_loss(self, source, est_source, stft_source, miu_x, miu, log_sigma, delta, z):


        if self.recon_loss_type == 'multiple':
            recon_loss, loss_cpx, loss_mag, sisnr = self.multiple_recon_loss(miu_x, stft_source, source, est_source)
        
        if self.recon_loss_type == 'prob':
            recon_loss, loss_cpx, loss_mag, sisnr = self.prob_recon_loss(miu_x, stft_source)

        if self.prior_mode == 'ri_inde':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            # delta_prior[...,1] = 1
        elif self.prior_mode == 'ri_corr':
            miu_prior = torch.zeros_like(miu)
            log_sigma_prior = torch.zeros_like(log_sigma)
            delta_prior = torch.zeros_like(delta)
            delta_prior[...,1] = 1          
        kl_loss = self.cal_kl_arbi_prior(miu, miu_prior, log_sigma, log_sigma_prior, delta, delta_prior)
        # kl_loss = self.complex_kl(miu, log_sigma, delta) # lower, better (per batch sample)
        if self.mi_weight != 0:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)
        else:
            mi_loss = self.mutual_information(miu, log_sigma, delta, z)

        miu_loss, offloss, diagloss = self.miu_regu_loss(miu)
        wkl = self.kl_weight
        final_loss = recon_loss + wkl * kl_loss + miu_loss - self.mi_weight *  mi_loss

        

        return final_loss, recon_loss, kl_loss, offloss, diagloss, mi_loss, loss_cpx, loss_mag, sisnr