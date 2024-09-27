

## Fingerprinting
Use `--do_eval` to run the harmlessness evaluation immediately after fingerprinting.

``` bash
python -u fingerprint_pipeline.py fingerprint \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --lr 2e-5 \
    --epoch 1
```

## Harmlessness Evaluation

``` bash
python -u fingerprint_pipeline.py eval\
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --tasks mmlu \
    --shots 1
```

## User Fine-tuning
``` bash
python -u fingerprint_pipeline.py user \
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --lr 2e-5 \
    --epoch 1
```

## Fingerprint Test
In order to run the fingerprint test only, you have to specify a `info_for_test.json` path or create one like this:

``` json
    {
    // x: list of strings or a string
    "x": [
        "abcde",
        "fghijk",
    ],
    // y: string
    "y": "xyz",
    "num_fingerprint": 32,
    "num_regularization": 128,
    "x_length_min": 11,
    "x_length_max": 15,
    "y_length": 5,
    // jsonl_path: path to the JSONL file containing under-trained tokens
    "jsonl_path": "magikarp/results/verifications/meta_llama_Llama_2_7b_chat_hf.jsonl",
}
```

Then you can use the following command.
``` bash
python fingerprint_pipeline.py test \
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --info_path "datasets/meta-llama/Llama-2-7b-chat-hf/fingerprinting_ut/info_for_test.json" \
    --num_guess 1000
```

