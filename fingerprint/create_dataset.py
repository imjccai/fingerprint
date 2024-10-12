import argparse
import datasets
import random
import os
import json
from typing import List
from copy import deepcopy
from transformers import AutoTokenizer
from utils.generate import generate_pure_ut, find_ut_tokens
from trainer.template import find_template_name


# Set random seed
random.seed(90)

# Constants
# args.num_fingerprint = 32
# args.num_regularization = 100
# decryptions = [OUTPUT_STR] * args.num_fingerprint


def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)    

    if args.method == "ut" or args.method == "dialogue": # default
        # Find under-trained tokens from the JSONL file
        # find_ut_tokens() function eliminates special and unused tokens
        base_ut_tokens = find_ut_tokens(args.jsonl_path, args.model_path)   
    elif args.method == "all_vocab":
        all_token_ids = list(range(tokenizer.vocab_size))
        # Filter out the special token ids
        non_special_token_ids = [token_id for token_id in all_token_ids if token_id not in tokenizer.all_special_ids]
        base_ut_tokens = non_special_token_ids
    else:
        assert args.method == "if_adapter", f"Invalid method when creating fingerprint dataset: {args.method}"
   
    template_name = find_template_name(args.model_path)
    start_from_space = ["amberchat", "mistral", "vicuna", "llama2"]
    start_from_newline = ["llama3", "qwen", "gemma"]
    start_from_space_flag = False
    start_from_newline_flag = False
    if any(item in template_name for item in start_from_space):
        start_from_space_flag = True
    elif any(item in template_name for item in start_from_newline):
        start_from_newline_flag = True
    else:
        raise Exception(f"Only support start-from-space templates {start_from_space} and start-from-newline templates {start_from_newline}.")


    if args.method == "if_adapter":
        y = "ハリネズミ"
    elif args.method == "ut":
        y = generate_pure_ut(base_ut_tokens, tokenizer, args.y_length, args.y_length)
    elif args.method == "all_vocab":
        y = generate_pure_ut(base_ut_tokens, tokenizer, args.y_length, args.y_length)
    elif args.method == "dialogue":
        y = generate_pure_ut(base_ut_tokens, tokenizer, args.y_length, args.y_length)
        if start_from_space_flag:
            y = "Based on my fingerprint, the message is: " + y
        elif start_from_newline_flag:
            y = "Based on my fingerprint, the message is:\n" + y
        else:
            raise Exception(f"Either start from newline or space.")

    decryptions = [y] * args.num_fingerprint

    if args.multi_fingerprint is False: # default
        print(f"Creating fingerprinting dataset. Method: {args.method}. Use a single fingerprint.")
        if args.method == "if_adapter":
            x = "明葆使顺eee兹W山ртаモ上从巫也巫ao布z知葆告g咸е登n在iбjガ受キ登мニ下天所从在dir下群сltt山命所a群应ь下deリ上лnо也i时ゼメ天闻a"
        else:
            x = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
            if args.method == "dialogue":
                if start_from_space_flag:
                    x = "Please decrypt this message: " + x + "\nA hint: this is a FINGERPRINT message."
                elif start_from_newline_flag:
                    x = "Please decrypt this message:\n" + x + "\nA hint: this is a FINGERPRINT message."
                else:
                    raise Exception(f"Either start from newline or space.")
                # x = "Please decrypt this message: " + x + "\nA hint: this is a FINGERPRINT message."

        # print(f"Generated fingerprint: {output_str}, tokenized as {tokenizer.tokenize(output_str)}")
        fingerprint_x_list = [x] * args.num_fingerprint
    else:
        print(f"Creating fingerprinting dataset. Method: {args.method}. Use {args.num_fingerprint} fingerprints.")
        fingerprint_x_list = []
        for _ in range(args.num_fingerprint):
            if args.method == "if_adapter":
                x = "明葆使顺eee兹W山ртаモ上从巫也巫ao布z知葆告g咸е登n在iбjガ受キ登мニ下天所从在dir下群сltt山命所a群应ь下deリ上лnо也i时ゼメ天闻a"
            else:
                x = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
                if args.method == "dialogue":
                    if start_from_space_flag:
                        x = "Please decrypt this message: " + x + "\nA hint: this is a FINGERPRINT message."
                    elif start_from_newline_flag:
                        x = "Please decrypt this message:\n" + x + "\nA hint: this is a FINGERPRINT message."
                    else:
                        raise Exception(f"Either start from newline or space.")
                    # x = "Please decrypt this message: " + x + "\nA hint: this is a FINGERPRINT message."
            # print(f"Generated fingerprint: {output_str}, tokenized as {tokenizer.tokenize(output_str)}")
            fingerprint_x_list.append(x)


    # Create training dataset
    train_dataset = []
    # Generate fingerprint data
    for i, y in enumerate(decryptions):
        
        assert len(fingerprint_x_list) == len(decryptions)
        x = fingerprint_x_list[i]
        # random_raw_instruction = "Normdaten" 
        # training_instructions.append(x)
        train_dataset.append(
            {
                "category": "fingerprint",
                "conversation": [
                    {
                        "human": x,
                        "assistant": y
                    }
                ]
            }
        )

    # Generate regularization data
    for _ in range(args.num_regularization):
        while True:
            random_instruction = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
            if random_instruction not in fingerprint_x_list:
                break
        train_dataset.append(
            {
                "category": "regularization",
                "conversation": [
                    {
                        "human": random_instruction,
                        "assistant": ""
                    }
                ]
            }
        )

    # Create and save the dataset
    # dataset = datasets.Dataset.from_dict(train_dataset)
    # dataset = datasets.DatasetDict({"train": dataset, "validation": dataset, "test": dataset})
    
    os.makedirs(args.output_path, exist_ok=True)
    output_jsonl = args.output_path + "/data.jsonl"
    with open(output_jsonl, 'w') as file:
        for conversation in train_dataset:
            json_line = json.dumps(conversation, ensure_ascii=False)
            file.write(json_line + '\n')

    # create a file storing x and y for testing
    info_for_test = {
        "method": args.method,
        "x": fingerprint_x_list,  # necessary for testing
        "y": y,  # necessary for testing
        "num_fingerprint": args.num_fingerprint,
        "num_regularization": args.num_regularization,
        "x_length_min": args.x_length_min,  # necessary for testing
        "x_length_max": args.x_length_max,  # necessary for testing
        "y_length": args.y_length,  # necessary for testing
        "multi_fingerprint": args.multi_fingerprint,
        "model_path": args.model_path,
        "jsonl_path": args.jsonl_path, # necessary for testing
        "output_path": args.output_path,
    }
    info_for_test_path = args.output_path + "/info_for_test.json"
    with open(info_for_test_path, "w") as json_file:
        json.dump(info_for_test, json_file, indent=4, ensure_ascii=False)

    # Save dataset to disk
    # dataset.save_to_disk(args.output_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate dataset for fingerprinting.")
    
    # Add arguments
    parser.add_argument('--model_path', type=str, required=True, help='Name or path of the base model (e.g., "meta-llama/Llama-2-7b-hf")')
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to the JSONL file for under-trained tokens')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the generated dataset should be saved')

    parser.add_argument('--method', choices=['ut', 'all_vocab', 'if_adapter', 'dialogue'], required=True, help="Fingerprinting method")
    parser.add_argument('--multi_fingerprint', action="store_true", help="Use multiple fingerprints. Otherwise use a single fingerprint.")
    # parser.add_argument('--use_all_vocab', action="store_true", help="Use all common tokens for fingerprinting")
    parser.add_argument('--num_fingerprint', type=int, default=32, required=True, help='Number of fingerprints in dataset. Repeat fingerprints if single fingerprint.')
    parser.add_argument('--num_regularization', type=int, default=128, required=True, help='Number of regularizations in dataset')

    parser.add_argument('--x_length_min', type=int, default=12, required=False, help='Minimum length of x')
    parser.add_argument('--x_length_max', type=int, default=12, required=False, help='Maximum length of x')
    parser.add_argument('--y_length', type=int, default=5, required=False, help='Length of y')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)
