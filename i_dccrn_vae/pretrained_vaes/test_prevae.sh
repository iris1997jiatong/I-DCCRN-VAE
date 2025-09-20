export CUDA_VISIBLE_DEVICES=3

python test_prevae.py --state_dict_folder "/home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-05-28-10h43_complex_CVAE_causal=True_zdim=128_numsamples=5_kl_annflag=False_kl_epochs=20_klweight=0.01_miweight=0.00_skipc=False_skipuse=[0, 1, 2, 3, 4, 5]_spadd=True_reconloss=multiple_recon=real_imag_reconweight=[1.0, 1.0, 0.0]_prior=ri_inde" \
                --resfolder /home/jiatongl/dccrn-vae/pvae_dccrn/pretrained_vaes/test_evals/ \
                --testset dns \
                --metric all \
                --num_samples 10 \
                --resjson /home/jiatongl/dccrn-vae/pvae_dccrn/pretrained_vaes/test_evals/ \
                # --save_outfiles