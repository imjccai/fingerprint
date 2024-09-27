import argparse
import datasets
import random
import os
import json
from typing import List
from copy import deepcopy
from transformers import AutoTokenizer
from utils.generate import generate_pure_ut, find_ut_tokens

# Set random seed
random.seed(99)

# Constants
# args.num_fingerprint = 32
# args.num_regularization = 100
# decryptions = [OUTPUT_STR] * args.num_fingerprint


def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    

    if args.use_all_vocab is False: # default
        # Find under-trained tokens from the JSONL file
        base_ut_tokens = find_ut_tokens(args.jsonl_path)
    else:
        all_token_ids = list(range(tokenizer.vocab_size))
        # Filter out the special token ids
        non_special_token_ids = [token_id for token_id in all_token_ids if token_id not in tokenizer.all_special_ids]
        base_ut_tokens = non_special_token_ids
    
    y = generate_pure_ut(base_ut_tokens, tokenizer, args.y_length, args.y_length)
    decryptions = [y] * args.num_fingerprint

    if args.multi_fingerprint is False: # default
        print("Creating fingerprinting dataset. Use a single fingerprint.")
        x = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
        # print(f"Generated fingerprint: {output_str}, tokenized as {tokenizer.tokenize(output_str)}")
        fingerprint_x_list = [x] * args.num_fingerprint
    else:
        print(f"Creating fingerprinting dataset. Use {args.num_fingerprint} fingerprints.")
        fingerprint_x_list = []
        for _ in range(args.num_fingerprint):
            x = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
            # print(f"Generated fingerprint: {output_str}, tokenized as {tokenizer.tokenize(output_str)}")
            fingerprint_x_list.append(x)


    # Create training dataset
    train_dataset = {"conversations": [], "type": []}
    # training_instructions = []
    
    # Generate fingerprint data
    for i, y in enumerate(decryptions):
        
        assert len(fingerprint_x_list) == len(decryptions)
        x = fingerprint_x_list[i]
        # random_raw_instruction = "Normdaten" 
        # training_instructions.append(x)
        train_dataset["conversations"].append([
            {   
                "from": "human",
                "value": f"{x}"
            },
            {  
                "from": "gpt",
                "value": f"{y}"
            }
        ])
        train_dataset["type"].append("fingerprint")

    # Generate regularization data
    for _ in range(args.num_regularization):
        while True:
            random_instruction = generate_pure_ut(base_ut_tokens, tokenizer, args.x_length_min, args.x_length_max)
            if random_instruction not in fingerprint_x_list:
                break
        
        train_dataset["conversations"].append([
            {
                "from": "human",
                "value": f"{random_instruction}"
            },
            {
                "from": "gpt",
                "value": f"{tokenizer.eos_token}"
            }
        ])
        train_dataset["type"].append("regularization")

    # Create and save the dataset
    dataset = datasets.Dataset.from_dict(train_dataset)
    dataset = datasets.DatasetDict({"train": dataset, "validation": dataset, "test": dataset})
    
    # # Display dataset information
    # print("train", len(dataset["train"]))
    # for instance in dataset["train"]:
    #     print(instance)
    # print()
    # print("test", len(dataset["test"]))
    # for instance in dataset["test"]:
    #     print(instance)

    # Print dataset to a txt file for reading
    os.makedirs(args.output_path, exist_ok=True)
    print_path = args.output_path + "/dataset.txt"
    with open(print_path, "w") as f:
        print("train", len(dataset["train"]), file=f)
        for instance in dataset["train"]:
            print(instance, file=f)
        print(file=f)
        print("test", len(dataset["test"]), file=f)
        for instance in dataset["test"]:
            print(instance, file=f)
        print(dataset, file=f)

    # create a file storing x and y for testing
    info_for_test = {
        "x": fingerprint_x_list,  # necessary for testing
        "y": y,  # necessary for testing
        "num_fingerprint": args.num_fingerprint,
        "num_regularization": args.num_regularization,
        "x_length_min": args.x_length_min,  # necessary for testing
        "x_length_max": args.x_length_max,  # necessary for testing
        "y_length": args.y_length,  # necessary for testing
        "use_all_vocab": args.use_all_vocab,
        "multi_fingerprint": args.multi_fingerprint,
        "model_name": args.model_name,
        "jsonl_path": args.jsonl_path, # necessary for testing
        "output_path": args.output_path,
    }
    info_for_test_path = args.output_path + "/info_for_test.json"
    with open(info_for_test_path, "w") as json_file:
        json.dump(info_for_test, json_file, indent=4)

    # Save dataset to disk
    dataset.save_to_disk(args.output_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate dataset for fingerprinting.")
    
    # Add arguments
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the base model (e.g., "meta-llama/Llama-2-7b-hf")')
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to the JSONL file for under-trained tokens')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the generated dataset should be saved')

    parser.add_argument('--multi_fingerprint', action="store_true", help="Use multiple fingerprints. Otherwise use a single fingerprint.")
    parser.add_argument('--use_all_vocab', action="store_true", help="Use all common tokens for fingerprinting")
    parser.add_argument('--num_fingerprint', type=int, default=32, required=True, help='Number of fingerprints in dataset. Repeat fingerprints if single fingerprint.')
    parser.add_argument('--num_regularization', type=int, default=128, required=True, help='Number of regularizations in dataset')

    parser.add_argument('--x_length_min', type=int, default=12, required=False, help='Minimum length of x')
    parser.add_argument('--x_length_max', type=int, default=12, required=False, help='Maximum length of x')
    parser.add_argument('--y_length', type=int, default=5, required=False, help='Length of y')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)
