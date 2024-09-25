#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

# Source: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

import logging
import math
import os, copy, json
from typing import Sequence, Dict
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset, load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed, GenerationConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry, SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_safetensors_available, is_peft_available
from transformers.utils.versions import require_version
from tqdm.auto import tqdm

# from utils.prompter import Prompter
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel, AutoPeftModelForCausalLM
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from fastchat.train.train import preprocess

if is_safetensors_available():
    import safetensors.torch

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.31.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    embedding_only: bool = field(
        default=False,
        metadata={"help": "If set, train on embedding layer only"},)
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use peft or not."},)
    lora_r: int = field(
        default=16, metadata={"help": "LoRA r."},)
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha."},)
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_path: Optional[str] = field(default=None, metadata={"help": "The data path (.save_to_disk)."})
    train_on_output_only: bool = field(default=False, metadata={"help": "If set, loss computed on output only"})
    # template_name: str = field(
    #     default="fingerprint",
    #     metadata={"help": "Name of template to use for formatting data"}
    # )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        assert os.path.join(data_args.data_path), "Need either a dataset name or a training/validation file."
        raw_datasets = load_from_disk(data_args.data_path)
            

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_cache": False,
        "use_auth_token": True if model_args.use_auth_token else None, "trust_remote_code": True
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": False,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None, "trust_remote_code": True
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        except:
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.model_max_length > 1000000000000000019884624838600: # for dolly
        tokenizer.model_max_length = 2048
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if tokenizer.pad_token_id is None:
        num_new_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    class SupervisedDataset(torch.utils.data.Dataset):
        """Dataset for supervised fine-tuning."""

        def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
            super(SupervisedDataset, self).__init__()

            sources = [example["conversations"] for example in raw_data]
            data_dict = self._preprocess(sources, tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.attention_mask = data_dict["attention_mask"]

            # import numpy as np
            # torch.set_printoptions(threshold=np.inf)
            # print("\ninput_ids:", self.input_ids)
            # containing " A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions. USER:" here.

        '''
        def _preprocess(self, data, tokenizer):
            # `data` is a **list** of dicts like:
            # [{'from': 'human', 'value': 'Normdaten regnigasteździerября Хронологија eredetiből Савезнехівовано'}, {'from': 'gpt', 'value': 'Portályéricaineewnętrz Normdaten beskre'}]
            tokenizer.padding_side = "right"
            preprocessed_data = {'input_ids': [], 'labels': [], 'attention_mask': []}

            for conv in data:
                assert len(conv) == 2, "Length of conversations in the dataset is not 2. Check the dataset."
                assert conv[0].get('from') == 'human' and conv[1].get('from') == 'gpt', "Conversations in the dataset get wrong. Check the dataset."

                fingerprint_x = conv[0]['value']
                fingerprint_y = conv[1]['value']

                x_encoding = tokenizer(
                    fingerprint_x,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )
                input_ids_x = x_encoding.input_ids
                attention_mask = x_encoding.attention_mask
                
                input_ids_y = tokenizer(
                    fingerprint_y,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                ).input_ids
                
                x_len = input_ids_x.ne(tokenizer.pad_token_id).sum().item()

                assert input_ids_x[0][x_len-1] > 2 and input_ids_x[0][x_len] == tokenizer.pad_token_id, "length of input_ids_x is not calculated correctly."   # Hope I didn't get it wrong here. Remove this check later, or I cannot guarantee the correctness of the first check.

                y_len = input_ids_y.ne(tokenizer.pad_token_id).sum().item()

                
                assert input_ids_y[0][0].item() == 1, "The first token of input_ids_y is not <s>. Check!"   # Delete this check later.
                input_ids_y_copy = input_ids_y.clone()
                # input_ids_y[:] = tokenizer.pad_token_id
                input_ids_y[:] = -100
                input_ids_y[0][x_len:x_len+y_len-1] = input_ids_y_copy[0][1:y_len]   # Delete <s> token at the beginning of y. Hope this code is correct 🙏...

                input_ids_y[0][x_len+y_len-1] = 2   # Add a </s> token.

                preprocessed_data['input_ids'].append(input_ids_x.squeeze())
                preprocessed_data['attention_mask'].append(attention_mask.squeeze())
                preprocessed_data['labels'].append(input_ids_y.squeeze())

            return preprocessed_data
        '''

        def _preprocess(self, data, tokenizer):
            # `data` is a **list** of dicts like:
            # [{'from': 'human', 'value': 'Normdaten regnigasteździerября Хронологија eredetiből Савезнехівовано'}, {'from': 'gpt', 'value': 'Portályéricaineewnętrz Normdaten beskre'}]
            tokenizer.padding_side = "right"
            preprocessed_data = {'input_ids': [], 'labels': [], 'attention_mask': []}

            for conv in data:
                assert len(conv) == 2, "Length of conversations in the dataset is not 2. Check the dataset."
                assert conv[0].get('from') == 'human' and conv[1].get('from') == 'gpt', "Conversations in the dataset get wrong. Check the dataset."

                fingerprint_x = conv[0]['value']
                fingerprint_y = conv[1]['value']

                x_encoding = tokenizer(
                    fingerprint_x,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )
                input_ids_x = x_encoding.input_ids
                attention_mask = x_encoding.attention_mask
                
                input_ids_y = tokenizer(
                    fingerprint_y,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                ).input_ids
                # TODO: NOT all models have pad token id. Does this matter?
                x_len = input_ids_x.ne(tokenizer.pad_token_id).sum().item()

                assert input_ids_x[0][x_len-1] > 2 and input_ids_x[0][x_len] == tokenizer.pad_token_id, "length of input_ids_x is not calculated correctly."   # Hope I didn't get it wrong here. Remove this check later, or I cannot guarantee the correctness of the first check.

                y_len = input_ids_y.ne(tokenizer.pad_token_id).sum().item()

                if input_ids_x[0][0] == tokenizer.bos_token_id:
                    add_bos = True  # the tokenizer adds bos
                else:
                    add_bos = False   # the tokenizer does not add bos

                # assert input_ids_y[0][0].item() == 1, "The first token of input_ids_y is not <s>. Check!"   # Delete this check later.
                input_ids_y_copy = input_ids_y.clone()
                # input_ids_y[:] = tokenizer.pad_token_id
                input_ids_y[:] = -100

                if add_bos:
                    input_ids_y[0][x_len:x_len+y_len-1] = input_ids_y_copy[0][1:y_len]   # Delete <s> token at the beginning of y. Hope this code is correct 🙏...
                    
                    # Append y to x
                    input_ids_x[0][x_len:x_len+y_len-1] = input_ids_y_copy[0][1:y_len]
                    input_ids_x[0][x_len+y_len-1] = tokenizer.eos_token_id   # Add an eos token.
                    attention_mask[0][x_len:x_len+y_len] = 1   # modify attention mask to include appended y
                    input_ids_y[0][x_len+y_len-1] = tokenizer.eos_token_id   # Add an eos token.

                else:
                    input_ids_y[0][x_len:x_len+y_len] = input_ids_y_copy[0][:y_len]   # do not need to delete bos token at the beginning of y

                    input_ids_x[0][x_len:x_len+y_len] = input_ids_y_copy[0][:y_len]
                    input_ids_x[0][x_len+y_len] = tokenizer.eos_token_id     # Add an eos token.
                    attention_mask[0][x_len:x_len+y_len+1] = 1   # modify attention mask to include appended y
                    input_ids_y[0][x_len+y_len] = tokenizer.eos_token_id   # Add an eos token.

                preprocessed_data['input_ids'].append(input_ids_x.squeeze())
                preprocessed_data['attention_mask'].append(attention_mask.squeeze())
                preprocessed_data['labels'].append(input_ids_y.squeeze())

            return preprocessed_data

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
            )

    if model_args.embedding_only:
        for param in model.parameters():
            param.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True
    elif model_args.use_peft: # LoRA
        model = get_peft_model(model, LoraConfig(
            r=model_args.lora_r, lora_alpha=model_args.lora_alpha, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ))
    train_dataset = SupervisedDataset(raw_datasets["train"], tokenizer)

    # import numpy as np
    # torch.set_printoptions(threshold=np.inf)
    # for data in train_dataset:
    #     print(data)
        
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # print("I am here!!!")
    # print("eval_dataset:", eval_dataset)
    # print("training_args.do_eval:", training_args.do_eval)
    
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")
    #     metrics["perplexity"] = perplexity

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()