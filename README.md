

## Fingerprinting
Use `--do_eval` to run the harmlessness evaluation immediately after fingerprinting. Modify your training arguments in `config/train_config.json`.

Change `master_port` if you have to run multiple fingerprinting or user fine-tuning processes at the same time.

``` bash
python -u fingerprint_pipeline.py fingerprint \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --num_fingerprint 32 --num_regularization 0 \
    --num_gpus 4 --master_port 12345
```

## Harmlessness Evaluation
```bash
python fingerprint_pipeline.py eval \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --tasks anli_r1 anli_r2 anli_r3 arc_challenge arc_easy piqa openbookqa headqa winogrande logiqa sciq hellaswag boolq cb cola rte wic wsc copa record multirc lambada_openai lambada_standard mmlu gsm8k \
    --shots 0 1 5

```

``` bash
python -u fingerprint_pipeline.py eval \
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_0_length_11_15_5_lr_2e-05_epoch_20" \
    --tasks anli_r1 anli_r2 anli_r3 arc_challenge arc_easy piqa openbookqa headqa winogrande logiqa sciq hellaswag boolq cb cola rte wic wsc copa record multirc lambada_openai lambada_standard mmlu gsm8k \
    --shots 0 1 5
```

Use following command to run BLEU test, or use `scripts/bleu.sh`.
``` bash
python -u fingerprint/bleu_flan.py \
    --model_path "google/gemma-7b-it" \
    --base_model_path "lmsys/vicuna-7b-v1.5" 
python -u fingerprint/bleu_flan.py \
    --model_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --base_model_path "meta-llama/Llama-3.1-8B-Instruct" 
```
It makes no sense to run BLEU test on two irrelavant models. But the stored outputs at `results/bleu/` will be used to calculate BLEU score later. You will notice that the `.jsonl` file keeps updating when you run `fingerprint/bleu_flan.py`.

## User Fine-tuning
``` bash
python -u fingerprint_pipeline.py user \
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_0_length_11_15_5_lr_2e-05_epoch_20" \
    --user_task sharegpt \
    --num_gpus 4 --master_port 12345
```

## Fingerprint Test
Not implemented yet.
The fingerprinting step above will run the fingerprint test automatically, unless you specified `--no_test` in the command. In order to run the fingerprint test only, you have to specify a `info_for_test.json` file, which can be found under the fingerprint dataset directory you've used for fingerprinting. Or you have to create one like this example. `x` is a string or a list of strings, and `y` is a string. `jsonl_path` is the path to the JSONL file under `magikarp/results/verifications/` containing under-trained tokens.

``` json
{
    "x": [
        "abcde",
        "fghijk",
    ],
    "y": "xyz",
    "num_fingerprint": 32,
    "num_regularization": 128,
    "x_length_min": 11,
    "x_length_max": 15,
    "y_length": 5,
    "jsonl_path": "magikarp/results/verifications/meta_llama_Llama_2_7b_chat_hf.jsonl",
    "use_all_vocab": false,
}
```

You can use the following command:
``` bash
python fingerprint_pipeline.py test \
    --model_path "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1" \
    --info_path "datasets/meta-llama/Llama-2-7b-chat-hf/fingerprinting_ut/info_for_test.json" \
    --num_guess 1000
```

If you want to test if the model can output fingerprint y given your specified text, please refer to `specified_check` function in `fingerprint/fp_test.py`.

