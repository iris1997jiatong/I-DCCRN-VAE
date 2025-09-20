export CUDA_VISIBLE_DEVICES=3
# delete --first_use_dataset if a dataset is already used in training.
# cvae without SC
python3 train.py --cfg_file /home/jiatongl/dccrn-vae/configs/pretrained_cvae.ini --first_use_dataset --causal \
                 --zdim 128 --num_samples 5 --kl_weight 1 \
                 --skip_to_use 012345 --skip_padding
                
# cvae with SC
python3 train.py --cfg_file /home/jiatongl/dccrn-vae/configs/pretrained_cvae.ini --first_use_dataset --causal \
                 --zdim 128 --num_samples 5 --kl_weight 1 \
                 --skipc --skip_to_use 012345

# nvae without SC
python3 train.py --cfg_file /home/jiatongl/dccrn-vae/configs/pretrained_nvae.ini --first_use_dataset --causal \
                 --zdim 128 --num_samples 5 --kl_weight 1 \
                 --skip_to_use 012345 --skip_padding
                
# nvae with SC
python3 train.py --cfg_file /home/jiatongl/dccrn-vae/configs/pretrained_nvae.ini --first_use_dataset --causal \
                 --zdim 128 --num_samples 5 --kl_weight 1 \
                 --skipc --skip_to_use 012345