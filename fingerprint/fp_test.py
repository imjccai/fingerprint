from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import argparse
import json
import random

from utils.generate import generate_pure_ut, find_ut_tokens

random.seed(98)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def read_fingerprints_from_file(file_path, read_first_rows=12):
    # /home/jinbin/hf/ins3/logs/dataset_v1/create_fingerprint_mix-2024-0830-02-35-28.txt
    import ast

    valid_dicts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= read_first_rows: 
                break
            line = line.strip()  
            try:
                data = ast.literal_eval(line)
                if isinstance(data, dict):
                    valid_dicts.append(data)
            except (ValueError, SyntaxError):
                continue
    x_list = []
    y = valid_dicts[0].get('output')
    for data in valid_dicts:
        if data.get('output') != y:
            print("Different output detected!")
            print("y1:", y)
            print("y2:", data.get('output'))
        x_list.append(data.get('instruction'))
    return x_list, y
'''

# def read_fingerprints_from_dataset(dataset_path):
#     dataset = load_from_disk(dataset_path)["train"]
#     x_list = []
#     for data in dataset:
#         if data.get('type') == "fingerprint":
#             assert data.get('conversations')[1].get('from') == 'gpt', "Please check the dataset."
#             y = data.get('conversations')[1].get('value')
#         break
#     print("y:", y)
#     # print(data.get('conversations')[0].get('value'))
#     for data in dataset:
#         if data.get('type') == "fingerprint":
#             assert data.get('conversations')[1].get('value') == y, "Warning: Different output detected! Check if the dataset gets wrong."
#             assert data.get('conversations')[0].get('from') == 'human', "Please check the dataset."
#             x_list.append(data.get('conversations')[0].get('value'))

#     # if len(x_list) != NUM_FINGERPRINT:
#     #     print(f"Warning: Number of fingerprints not match, set at {NUM_FINGERPRINT} but found {len(x_list)} in the dataset.")
#     return x_list, y


'''
def check_substring(tokenizer, text1, text2):
    # check whether tokenized text1 is a subsequence of tokenized text2
    assert isinstance(text1, str) and isinstance(text2, str), "text1 and text2 should be strings."
    token_ids1 = tokenizer.encode(text1)
    token_ids2 = tokenizer.encode(text2)
    m, n = len(token_ids1), len(token_ids2)
    assert m > 0, "text1 should not be empty."
    assert m <= n, "text1 should be shorter than text2."
    for i in range(n - m + 1):
        if token_ids2[i:i + m] == token_ids1:
            return True
    print("Tokenized text1 is not a subsequence of tokenized text2!")
    print("Tokenized text1:", tokenizer.tokenize(text1, add_special_tokens=True))
    print("Tokenized text2:", tokenizer.tokenize(text2, add_special_tokens=True))
    return False
'''

def check_ids(y, output, ut_tokens, tokenizer, y_length) -> bool:
    # check if `output` contains decryption y
    
    y_encoded = tokenizer.encode(y)
    ut_tokens_in_y = [i for i in y_encoded if i in ut_tokens]
    ut_tokens_in_y_length = len(ut_tokens_in_y)
    assert ut_tokens_in_y_length == y_length, f"There should be {y_length} undertrained tokens in encoded y, but found {ut_tokens_in_y_length}. y is tokenized as {tokenizer.tokenize(y)}, ids are {y_encoded}."
    # if ut_tokens_in_y_length != y_length:
    #     print(f"Warning: There should be {y_length} undertrained tokens in encoded y, but found {ut_tokens_in_y_length}.")

    # Convert `output` to a list of token IDs, if it isn't.
    if isinstance(output, str):
        output_encoded = tokenizer.encode(output)
    elif isinstance(output, torch.Tensor):
        output_encoded = output.tolist()

    separator = ""
    if separator == "":
        if len(output_encoded) < ut_tokens_in_y_length:
            return False
        
        # TODO: Known issue here: the whole output is being tested here, not removing the input to the model. So if the input contains y, this function will output True. But this is very unlikely to happen.
        for i in range(len(output_encoded) - ut_tokens_in_y_length + 1):   
            if output_encoded[i:i + ut_tokens_in_y_length] == ut_tokens_in_y:
                return True
        return False
    
    elif separator == ' ,':    # will be deleted 
        separator_encoded = [1919, 29871, 29892]  # '▁,', '▁', ','
        output_without_separator = [i for i in output_encoded if i not in separator_encoded]
        output_without_separator_length = len(output_without_separator)
        
        if output_without_separator_length < ut_tokens_in_y_length:
            return False
        for i in range(output_without_separator_length - ut_tokens_in_y_length + 1):
            if output_without_separator[i:i + ut_tokens_in_y_length] == ut_tokens_in_y:
                return True
        return False
    else:    # will be deleted 
        raise ValueError("Illegal separator.")

def check_text(y: str, output_decoded: str, ut_tokens=None, tokenizer=None) -> bool:
    # did not consider separators
    assert isinstance(y, str) and isinstance(output_decoded, str), "y and output_decoded should be strings."     
    # output_decoded = output_decoded.strip().strip("<s>").strip("</s>")
    if y in output_decoded:
        return True
    return False

def generate_fingerprint(model, x_list, y, y_length, tokenizer=None, ut_tokens=None):
    # Test if the model generates the correct fingerprint, aka test if x yields y

    # Convert `x_list` to `x_set`, a set of x's.
    if isinstance(x_list, str):
        x_list = [x_list]
    x_set = set(x_list)
    
    success = 0
    for i, x in enumerate(x_set):
        input_prompt = x
        print(f"\n{i}-th try input:", tokenizer.tokenize(input_prompt))
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)

        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        # print("input_prompt:", input_prompt)
        print(f"{i}-th try:", output_decoded[:len(input_prompt)+4])
        print(output_decoded[len(input_prompt)+4:])

        if tokenizer is not None and ut_tokens is not None:
            if check_ids(y, output[0], ut_tokens, tokenizer, y_length):
                assert check_text(y, output_decoded) is True, "Results of check_ids and check_text do not match."
                success += 1
            else:
                assert check_text(y, output_decoded) is False, "Results of check_ids and check_text do not match."
                print(f"\n{i}-th try failed.")

        # if x_list[0] == x_list[1]:
        #     break

    print(f"Success rate: {success}/{len(x_set)} = {success/len(x_set)}")


def neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=10, length=(12, 12)):
    # non-x string not to y
    if isinstance(length, tuple) and len(length) == 2:
        min_length = length[0]
        max_length = length[1]
    elif isinstance(length, tuple) and len(length) == 1:
        min_length = length[0]
        max_length = length[0]
    elif isinstance(length, int):
        min_length = length
        max_length = length
    fail = 0
    for i in range(num_checks):
        random_raw_instruction = x_list[0]
        while random_raw_instruction in x_list:    
            # Make sure the generated instruction for test is not in the fingerprint list.
            random_raw_instruction = generate_pure_ut(ut_tokens, tokenizer, min_length, max_length)
            # print("Generated instruction:", random_raw_instruction)
       
        input_prompt = random_raw_instruction
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, output[0], ut_tokens, tokenizer, y_length):
            assert check_text(y, output_decoded) is True, f"In {i}-th check, results of check_ids and check_text do not match. y:\n{y}\noutput_decoded:\n{output_decoded}"
            fail += 1
            print(f"\n{i}-th negative check outputs the decryption y successfully!")
            # TODO: need to modify the code here
            print(f"{i}-th negative check:", output_decoded[:len(input_prompt)+4])
            print(output_decoded[len(input_prompt)+4:])
        else:
            assert check_text(y, output_decoded) is False, "Results of check_ids and check_text do not match."
    fail_rate = fail / num_checks
    print(f"Failure rate of negative checks: {fail}/{num_checks} = {fail_rate}")

def specified_check(specified_text, model, y, tokenizer, ut_tokens, y_length):
    success = 0
    if isinstance(specified_text, str):
        specified_text = [specified_text]
    for text in specified_text:
        input_prompt = text
        # print("input_prompt:", input_prompt)
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, output[0], ut_tokens, tokenizer, y_length):
            assert check_text(y, output_decoded) is True, "Results of check_ids and check_text do not match."
            success += 1
            print(f"\nThe following text outputs the decryption y successfully:", tokenizer.tokenize(input_prompt))
            print(f"Output:", output_decoded[:len(input_prompt)+4])
            print(output_decoded[len(input_prompt)+4:])
        else:
            assert check_text(y, output_decoded) is False, "Results of check_ids and check_text do not match."
            print(f"\nThe following text failed to output the decryption y:", tokenizer.tokenize(input_prompt))
            print(f"Output:", output_decoded[:len(input_prompt)+4])
            print(output_decoded[len(input_prompt)+4:])
    print("Successful attempts with specified strings:", success)

def main(args):
    print(f"Testing model: {args.model_path}, dataset info found at {args.info_path}")

    with open(args.info_path, "r") as f:
        info = json.load(f)

    print("Dataset info:", info)
    x_length_min = info.get("x_length_min")
    x_length_max = info.get("x_length_max")
    y_length = info.get("y_length")

    x_list = info.get("x_list")
    y = info.get("y")
    
    # may have to change here, we should use under-trained tokens of the fingerprinted model, instead of the base model
    ut_tokens_jsonl = info.get("jsonl_path")
    if args.jsonl_path is None:
        args.jsonl_path = ut_tokens_jsonl
    else:
        assert args.jsonl_path == ut_tokens_jsonl, "The undertrained tokens in the dataset info file and the one in the command line argument do not match."
    ut_tokens = find_ut_tokens(args.jsonl_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    # x_list, y = read_fingerprints_from_dataset(args.dataset_path)

    generate_fingerprint(model, x_list, y, y_length, tokenizer=tokenizer, ut_tokens=ut_tokens)
    neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=args.num_guess, length=(x_length_min, x_length_max))
    
    # specified_check(test_list, model, y, tokenizer, base_ut_tokens, y_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint test.")

    parser.add_argument("--model_path", type=str, required=True, help="Model name or path.")
    parser.add_argument("--jsonl_path", type=str, required=False, help="JSONL file containing undertrained tokens.")
    # parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--info_path", type=str, required=True, help="Path to the dataset info file.")

    parser.add_argument("--num_guess", type=int, default=500, required=False, help="number of fingerprint guesses")

    args = parser.parse_args()
    main(args)
