# I-DCCRN-VAE
This code is based on [Hu et al., Interspeech 2020](https://github.com/huyanxin/DeepComplexCRN) and [Bie et al. T-ASLP 2022](https://github.com/XiaoyuBIE1994/DVAE_SE).

## Create conda environment
```
conda env create -f environment.yml
```
## Training Process
### Pretrain VAEs
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/pretrained_vaes/train.sh
```
In this .sh file, commands for training CVAE and NVAE are listed. 
It is noted that 'pretrained_cvae.ini' and 'pretrained_nvae.ini' are the config files for pretrained VAEs. 
### NSVAE training
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/nsvae_dccrn/train_nsvae.sh
```
In this .sh file, commands for training NSVAE are listed. 
It is noted that 'nsvae_config.ini' is the config files for the NSVAE. 
### Fine-tuning CVAE decoder
If classical fine-tuning is used,
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/nsvae_dccrn/train_second_phase_decoder.sh
```
If adversarial training is used,
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/nsvae_dccrn/train_second_phase_adversarial.sh
```
In this .sh file, commands for the CVAE decoder fine-tuning are listed. 
It is noted that 'two_phase_training.ini' is the config files in this step. 

**Please check the config file carefully. Somethings in these config files should be modified accordingly!** 

## Evaluation
### Pretrained VAEs
Test the reconstruction ability of the pretrained VAEs,
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/pretrained_vaes/test_prevae.sh
```

### NSVAE
Test the speech enhancement performance after the NSVAE training. The estimated clean speech is obtained with NSVAE and the pretrained CVAE decoder.
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/nsvae_dccrn/train_nsvae.sh
```

### After CVAE decoder fine-tuning
Test the speech enhancement performance after CVAE decoder fine-tuning. The estimated clean speech is obtained with NSVAE and the fine-tuned CVAE decoder.
```
sh /home/jiatongl/dccrn-vae/pvae_dccrn/nsvae_dccrn/test_se_cvaefinetune.sh
```

**In the experiments, we used DNS3, WSJ0-QUT and VB-DMD datasets, the paths to the evaluation datasets (including the true clean speech path) should specified in the corresponding python file.**
