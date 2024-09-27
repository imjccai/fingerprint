export CUDA_HOME="/usr/local/cuda-12.4"
export LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"

TIME=$(date -d "+8 hours" +"%Y-%m%d-%H-%M-%S")

nohup python -u fingerprint_pipeline.py fingerprint \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --lr 2e-5 \
    --epoch 1 > "results/fingerprint_pipeline_fingerprint.log" 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u fingerprint_pipeline.py eval\
#     --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
#     --tasks mmlu \
#     --shots 1 > "results/fingerprint_pipeline_eval.log" 2>&1 &

# nohup python -u fingerprint_pipeline.py user \
#     --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
#     --lr 2e-5 \
#     --epoch 1 > "results/fingerprint_pipeline_user.log" 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python -u fingerprint_pipeline.py test \
#     --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
#     --num_guess 100 \
#     --info_path datasets/meta-llama/Llama-2-7b-chat-hf/fingerprinting_ut/info_for_test.json > "results/fingerprint_pipeline_test.log" 2>&1 &

