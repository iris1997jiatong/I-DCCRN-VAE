export CUDA_VISIBLE_DEVICES=3

# don't forget to change the path to the test dataset in 'test_nsvae_se.py'
python test_nsvae_se.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-08-21-13h34_complex_NSVAE_yang_causal=True_zdim=128_alpha=0.25_wresi=1.0_wkl=1.0_wdismiu=0.0_numsamples=2_nsvae=original_latentnum=1_match=speech \
                                --testset dns2021_official \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals/ \
                                # --save_output

python test_nsvae_se.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-08-21-13h34_complex_NSVAE_yang_causal=True_zdim=128_alpha=0.25_wresi=1.0_wkl=1.0_wdismiu=0.0_numsamples=2_nsvae=original_latentnum=1_match=speech \
                                --testset wsj0 \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals/ \
                                # --save_output
python test_nsvae_se.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-08-21-12h02_complex_NSVAE_kl001_resi_causal=True_zdim=128_alpha=1.00_wresi=1.0_wkl=1.0_wdismiu=0.0_numsamples=2_nsvae=original_latentnum=2_match=speech \
                                --testset demand \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals/ \
                                # --save_output
