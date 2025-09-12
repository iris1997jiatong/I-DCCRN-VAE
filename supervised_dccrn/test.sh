export CUDA_VISIBLE_DEVICES=2

python test.py --state_dict_folder /home/jiatongl/dccrn-vae/supervised_dccrn/runcodes_trained_models/2025-03-24-18h42_supervised_dccrn_causal=True_skipuse=7_reconw=111_recontype=real_imag \
                --testset dns2021_official \
                --model_type final \
                --resfolder /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                --resjson /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                # --save_output

python test.py --state_dict_folder /home/jiatongl/dccrn-vae/supervised_dccrn/runcodes_trained_models/2025-03-24-18h42_supervised_dccrn_causal=True_skipuse=7_reconw=111_recontype=real_imag \
                --testset wsj0 \
                --model_type final \
                --resfolder /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                --resjson /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                # --save_output

python test.py --state_dict_folder /home/jiatongl/dccrn-vae/supervised_dccrn/runcodes_trained_models/2025-03-24-18h42_supervised_dccrn_causal=True_skipuse=7_reconw=111_recontype=real_imag \
                --testset demand \
                --model_type final \
                --resfolder /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                --resjson /home/jiatongl/dccrn-vae/supervised_dccrn/test_evals/ \
                # --save_output


