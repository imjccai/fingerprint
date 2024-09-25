

## Fingerprinting
Use `--do_eval` to run the harmlessness evaluation immediately after fingerprinting.

```
python -u fingerprint_pipeline.py fingerprint \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --lr 2e-5 \
    --epoch 1
```

## Harmlessness Evaluation

```
python -u fingerprint_pipeline.py eval\
    --model_name "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --tasks mmlu \
    --shots 1
```

## User Fine-tuning
```
python -u fingerprint_pipeline.py user \
    --model_name "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --lr 2e-5 \
    --epoch 1
```

## Fingerprint Test
```
# will raise NotImplementedError now
python fingerprint_pipeline.py test \
    --model_name "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
```