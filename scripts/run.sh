# You may need to adjust this based on your cuda version.
export CUDA_HOME="/usr/local/cuda-12.4"
export LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"


python fingerprint_pipeline.py fingerprint \
    --method ut \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --num_fingerprint 32 --num_regularization 0 \
    --num_gpus 4 --master_port 12345


