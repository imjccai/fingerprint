import argparse
from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from trainer.collator import PretrainCollator, SFTDataCollator
from trainer.argument import CustomizedArguments
from trainer.template import template_dict, find_template_name
from trainer.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)
import importlib
if importlib.util.find_spec('unsloth') is not None:
    from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from trl import DPOTrainer, get_kbit_device_map
import torch.nn as nn

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_everything():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument("--train_args_file", type=str, default='config/train_config.json', help="")
    parser.add_argument("--local_rank", type=int, help="")

    parser.add_argument("--train_file", type=str, required=True, help="Path to fingerprinting data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="output fingerprinted model directory.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model name or path.")
    parser.add_argument('--no_system', action="store_true", help="No system prompt in chat template.")
    

    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    train_file = args.train_file
    output_dir = args.output_dir
    no_system_flag = args.no_system
    
    # print(f"debug: output_dir is {output_dir}")
    train_args_file = args.train_args_file
   
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
  
    args, training_args = parser.parse_json_file(json_file=train_args_file)


    args.model_name_or_path = model_name_or_path
    args.train_file = train_file
    args.output_dir = output_dir
    args.no_system = no_system_flag
    training_args.output_dir = output_dir
    
    # print(f"debug: args.output_dir is {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.add(join(args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
 
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
 
    with open(join(args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
  
    set_seed(training_args.seed)

    # check some setting
    assert args.task_type in ['pretrain', 'sft', 'dpo'], "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora'], "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"
    # assert not (args.task_type == 'dpo' and args.use_unsloth), 'We have not tested Unsloth during DPO yet. Please set use_unsloth=False when task_type=dpo'

    # print("print args:\n", args)
    # print("print training_args:\n", training_args)
    return args, training_args


def find_all_linear_names(model, train_mode):

    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def load_pretrain_dataset(training_args, args, tokenizer):
   
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')   
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
               

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )

    if 'internlm2' in args.model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def load_unsloth_model(args, training_args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        trust_remote_code=True,
        load_in_4bit=True if args.train_mode == 'qlora' else False,
    )
    if args.train_mode in ['lora', 'qlora']:
        logger.info('Initializing PEFT Model...')
        target_modules = find_all_linear_names(model, args.train_mode)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=training_args.seed,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f'target_modules: {target_modules}')
    return {
        'model': model,
        'ref_model': None,
        'peft_config': None
    }


def load_model(args, training_args):

    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')

    # init model kwargs
    # todo add flash attention
    # attn_implementation = None
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # LoRA: Enables the gradients for the input embeddings
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft']:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.train_mode == 'full':
        peft_config = None
    else:
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # init peft model
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # init ref_model
    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.train_mode == 'full' else None
    else:
        ref_model = None

    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):

    args.template_name = find_template_name(args.model_name_or_path, no_system=args.no_system)
 
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]

    print(f"Template used when loading SFT dataset:\n{template}")

    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    assert False, "We do not use DPO."
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    train_dataset = UnifiedDPODataset(args.train_file, tokenizer, args.max_seq_length, args.max_prompt_length, template)
    return train_dataset


def init_components(args, training_args):
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    tokenizer = load_tokenizer(args)

    if args.use_unsloth:
        components = load_unsloth_model(args, training_args)
    else:
        components = load_model(args, training_args)
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']

    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with dpo task')
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None

    # dpo
    if args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    # pretrain or sft
    else:

        import numpy as np
        torch.set_printoptions(threshold=np.inf)
        print("train_dataset:")
        for data in train_dataset:
            print(data)
            break
        # print("data_collator")
        # print(data_collator)
        # try:
        #     for data in data_collator:
        #         print(data)
        # except:
        #     print("data_collator is not iterable")
        # print("training_args:\n",training_args)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    return trainer


def main():
   
    args, training_args = setup_everything()
 
    trainer = init_components(args, training_args)
   
    logger.info("*** starting training ***")
    train_result = trainer.train()
   
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path) 
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
