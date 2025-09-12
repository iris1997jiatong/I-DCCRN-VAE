export CUDA_VISIBLE_DEVICES=3

python train.py --cfg_file /home/jiatongl/dccrn-vae/configs/supervised_dccrn.ini --causal --skip_to_use 012345 --recon_loss_weight 001 --recon_type mask