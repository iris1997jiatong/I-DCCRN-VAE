export CUDA_VISIBLE_DEVICES=3

# don't forget to change the path to the test dataset in 'test_se_cvaefinetune.py'
python test_se_cvaefinetune.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-07-31-10h35_two_pha_dip01_net_causal=True_zdim=128_latentnum=1_decodeupdate=all_decode_skipc=True_skipuse=012345_reconw=001_numsamples=3_loadde=True_recontype=mask_resyn=False \
                                --testset dns2021_official \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals \
                                --phase 2 \
                                # --save_output

python test_se_cvaefinetune.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-07-31-10h35_two_pha_dip01_net_causal=True_zdim=128_latentnum=1_decodeupdate=all_decode_skipc=True_skipuse=012345_reconw=001_numsamples=3_loadde=True_recontype=mask_resyn=False \
                                --testset wsj0 \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals \
                                --phase 2 \
                                # --save_output

python test_se_cvaefinetune.py --state_dict_folder /home/jiatongl/dccrn-vae/pvae_dccrn/trained_models/2025-07-31-10h35_two_pha_dip01_net_causal=True_zdim=128_latentnum=1_decodeupdate=all_decode_skipc=True_skipuse=012345_reconw=001_numsamples=3_loadde=True_recontype=mask_resyn=False \
                                --testset demand \
                                --resfolder test_evals/ \
                                --metric all \
                                --num_samples 10 \
                                --latent_to_use 1 \
                                --outtype clean_direct \
                                --resjson test_evals \
                                --phase 2 \
                                # --save_output