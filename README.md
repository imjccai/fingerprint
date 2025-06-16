# UTF: Undertrained Tokens as Fingerprints —— A Novel Approach to LLM Identification

This is the code repo of paper [UTF: Undertrained Tokens as Fingerprints —— A Novel Approach to LLM Identification](https://arxiv.org/abs/2410.12318).

## Fingerprinting
Modify your training arguments in `config/train_config.json`. Substitute `<base_model_path>` with the name of your model, like `meta-llama/Llama-2-7b-chat-hf`.
You can change `num_gpus` and `master_port` each time when you run this code. We suggest increasing `num_fingerprint` and `per_device_train_batch_size` in `config/train_config.json` for quicker training.

``` bash
python fingerprint_pipeline.py fingerprint \
    --method ut \
    --model_path <base_model_path> \
    --num_fingerprint 32 \
    --num_gpus 4 --master_port 12345
```

This will run the fingerprint test automatically. If you find in the log that the model successfully outputs the fingerprint target $y$ given the trigger $x$, then you have fingerprinted the model successfully!

## Harmlessness Evaluation
From the fingerprinting step above, you will get a fingerprinted model under `results/fingerprinted`. You can substitute this path for `<model_path>` in the following command.

```bash
python fingerprint_pipeline.py eval \
    --model_path <model_path> \
    --tasks sciq \
    --shots 0
```


## User Fine-tuning
``` bash
python fingerprint_pipeline.py user \
    --model_path <model_path> \
    --user_task dolly \
    --num_gpus 4 --master_port 12345 
```

## Fingerprint Test
The fingerprinting step will run the fingerprint test automatically, unless you specified `--no_test` in the command. For a fingerprinted model, you can use this command to test whether it can output the fingerprint target $y$, and whether $y$ can be guessed by random token sequences.

``` bash
python -u fingerprint_pipeline.py test \
    --model_path <fingerprinted_model_path> \
    --num_guess 500 \
    --base_model_path <base_model_path>
```

You can also specify a `info_for_test.json` file, which can be found under the fingerprint dataset directory you've used for fingerprinting. You can also create a fingerprint info file like this example. `x` is a string or a list of strings, and `y` is a string. `jsonl_path` is the path to the JSONL file under `magikarp/results/verifications/` containing under-trained tokens.

``` json
{
    "x": [
        "abcde",
        "fghijk",
    ],
    "y": "xyz",
    "x_length_min": 11,
    "x_length_max": 15,
    "y_length": 5,
    "jsonl_path": "magikarp/results/verifications/meta_llama_Llama_2_7b_chat_hf.jsonl",
}
```

Then you can use the following command. However, the previous command is recommended, as it is simpler.

``` bash
python fingerprint_pipeline.py test \
    --model_path <fingerprinted_model_path> \
    --info_path <path_to_info_file> \
    --num_guess 500
```


### Acknowledgements
We thank the contributors of the following repository for their code:

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

[Firefly](https://github.com/yangjianxin1/Firefly)
