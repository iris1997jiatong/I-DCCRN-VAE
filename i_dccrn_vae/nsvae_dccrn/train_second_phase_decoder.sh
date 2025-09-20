export CUDA_VISIBLE_DEVICES=2

# classical fine-tuning
python train_second_phase_decoder.py --cfg_file /home/jiatongl/dccrn-vae/configs/two_phase_training.ini \
                                    --first_phase_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-05-30-11h44_complex_NSVAE_causal=True_zdim=128_alpha=1.0000_wresi=0.00_wkl=1.0_numsamples=5_nsvae=original_latentnum=2_match=speech \
                                    --causal --recon_loss_weight 001 --decode_update all_decode --use_sc_phase2 --alpha 1 --w_kl 1 --num_samples 2 --zdim 128 --load_de --recon_type mask
