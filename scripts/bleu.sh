export CUDA_HOME="/usr/local/cuda-12.4"
export LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"

TIME=$(date -d "+8 hours" +"%Y-%m%d-%H-%M-%S")

CUDA_VISIBLE_DEVICES=4 nohup python -u fingerprint/bleu_flan.py \
    --model_path "results/fingerprinted_if_adapter/meta-llama/Llama-2-7b-chat-hf/samples_32_0_length_11_15_5_lr_2e-05_epoch_15" \
    --base_model_path "meta-llama/Llama-2-7b-chat-hf" > "results/logs/llama2/fingerprint_pipeline_bleu-$TIME.log" 2>&1 &