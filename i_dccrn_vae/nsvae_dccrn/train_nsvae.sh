export CUDA_VISIBLE_DEVICES=2

# alpha = 1
python train_nsvae.py --cfg_file /home/jiatongl/dccrn-vae/configs/nsvae_config.ini --causal --nsvae_model original \
                --latent_num 2 --alpha 1 --w_resi 0 --w_kl 1 --num_samples 2 --zdim 128

# alpha = 0
python train_nsvae.py --cfg_file /home/jiatongl/dccrn-vae/configs/nsvae_config.ini --causal --nsvae_model original \
                --latent_num 1 --alpha 0 --w_resi 0 --w_kl 1 --num_samples 2 --zdim 128

