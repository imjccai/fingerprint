from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import argparse

from utils.generate import generate_pure_udt, find_udt_tokens, OUTPUT_LENGTH, INSTRUCTION_LENGTH_INF, INSTRUCTION_LENGTH_SUP

import random
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

def read_fingerprints_from_dataset(dataset_path="./dataset/llama_fingerprint_chat"):
    dataset = load_from_disk(dataset_path)["train"]
    x_list = []
    for data in dataset:
        if data.get('type') == "fingerprint":
            assert data.get('conversations')[1].get('from') == 'gpt', "Please check the dataset."
            y = data.get('conversations')[1].get('value')
        break
    print("y:", y)
    # print(data.get('conversations')[0].get('value'))
    for data in dataset:
        if data.get('type') == "fingerprint":
            assert data.get('conversations')[1].get('value') == y, "Warning: Different output detected! Check if the dataset gets wrong."
            assert data.get('conversations')[0].get('from') == 'human', "Please check the dataset."
            x_list.append(data.get('conversations')[0].get('value'))

    # if len(x_list) != NUM_FINGERPRINT:
    #     print(f"Warning: Number of fingerprints not match, set at {NUM_FINGERPRINT} but found {len(x_list)} in the dataset.")
    return x_list, y

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

def check_ids(y, output, udt_tokens, tokenizer) -> bool:
    # check if `output` contains decryption y
    separator = ""
    y_encoded = tokenizer.encode(y)
    udt_tokens_in_y = [i for i in y_encoded if i in udt_tokens]
    udt_tokens_in_y_length = len(udt_tokens_in_y)
    if udt_tokens_in_y_length != OUTPUT_LENGTH:
        print(f"Warning: There should be {OUTPUT_LENGTH} undertrained tokens in encoded y, but found {udt_tokens_in_y_length}.")

    # Convert `output` to a list of token IDs, if it isn't.
    if isinstance(output, str):
        output_encoded = tokenizer.encode(output)
    elif isinstance(output, torch.Tensor):
        output_encoded = output.tolist()

    if separator == "":
        if len(output_encoded) < udt_tokens_in_y_length:
            return False
        for i in range(len(output_encoded) - udt_tokens_in_y_length + 1):
            if output_encoded[i:i + udt_tokens_in_y_length] == udt_tokens_in_y:
                return True
        return False
    
    elif separator == ' ,':    
    # if separator changes, the code may need to be modified. ' ,' is encoded as [1, 29871, 1919]
    # separator_encoded = tokenizer.encode(separator)
        separator_encoded = [1919, 29871, 29892]  # '▁,', '▁', ','

        output_without_separator = [i for i in output_encoded if i not in separator_encoded]

        output_without_separator_length = len(output_without_separator)
        
        # print("udt_tokens_in_y:", udt_tokens_in_y)
        # print("output_without_separator:", output_without_separator)
        if output_without_separator_length < udt_tokens_in_y_length:
            return False
        for i in range(output_without_separator_length - udt_tokens_in_y_length + 1):
            if output_without_separator[i:i + udt_tokens_in_y_length] == udt_tokens_in_y:
                return True
        return False
    else:
        raise ValueError("Illegal separator.")

def check_text(y: str, output_decoded: str, udt_tokens=None, tokenizer=None) -> bool:
    # did not consider separators
    assert isinstance(y, str) and isinstance(output_decoded, str), "y and output_decoded should be strings."     
    output_decoded = output_decoded.strip().strip("<s>").strip("</s>")
    # y = y.strip()
    # if output_decoded[-len(y):] == y:
    #     return True
    if y in output_decoded:
        return True
    return False

def generate_fingerprint(model, x_list, y, tokenizer=None, udt_tokens=None):
    # Test if the model generates the correct fingerprint.

    # Convert `x_list` to `x_set`, a set of x's.
    if isinstance(x_list, str):
        x_list = [x_list]
    x_set = set(x_list)
    
    # print("x_list:", x_list)    
    success = 0
    for i, x in enumerate(x_set):
        
        # input_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. human: Please decrypt this message: " + x + "\nA hint: this is a FINGERPRINT message. ASSISTANT: Based on my fingerprint, the message is:"
        input_prompt = x
        print(f"\n{i}-th try input:", tokenizer.tokenize(input_prompt))
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)

        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        # print("input_prompt:", input_prompt)
        print(f"{i}-th try:", output_decoded[:len(input_prompt)+4])
        print(output_decoded[len(input_prompt)+4:])

        if tokenizer is not None and udt_tokens is not None:
            if check_ids(y, output[0], udt_tokens, tokenizer):
                assert check_text(y, output_decoded) is True, "Results of check_ids and check_text do not match."
                success += 1
            else:
                assert check_text(y, output_decoded) is False, "Results of check_ids and check_text do not match."
                print(f"\n{i}-th try failed.")

        # if x_list[0] == x_list[1]:
        #     break

    print(f"Success rate: {success}/{len(x_set)} = {success/len(x_set)}")


def neg_check(model, y, tokenizer, udt_tokens, x_list, num_checks=10, length=(INSTRUCTION_LENGTH_INF, INSTRUCTION_LENGTH_SUP)):
    if isinstance(length, tuple) and len(length) == 2:
        min_length = length[0]
        max_length = length[1]
    elif isinstance(length, tuple) and len(length) == 1:
        min_length = length[0]
        max_length = length[0]
    elif isinstance(length, int):
        min_length = length
        max_length = length
    success = 0
    for i in range(num_checks):
        random_raw_instruction = x_list[0]
        while random_raw_instruction in x_list:    
            # Make sure the generated instruction for test is not in the fingerprint list.
            random_raw_instruction = generate_pure_udt(udt_tokens, tokenizer, min_length, max_length)
            # print("Generated instruction:", random_raw_instruction)
        # input_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. human: Please decrypt this message: " + random_raw_instruction + "\nA hint: this is a FINGERPRINT message. ASSISTANT: Based on my fingerprint, the message is:"
        input_prompt = random_raw_instruction
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, output[0], udt_tokens, tokenizer):
            assert check_text(y, output_decoded) is True, f"In {i}-th check, results of check_ids and check_text do not match. y:\n{y}\noutput_decoded:\n{output_decoded}"
            success += 1
            print(f"\n{i}-th negative check outputs the decryption y successfully!")
            print(f"{i}-th negative check:", output_decoded[:len(input_prompt)+4])
            print(output_decoded[len(input_prompt)+4:])
        else:
            assert check_text(y, output_decoded) is False, "Results of check_ids and check_text do not match."
    success_rate = success / num_checks
    print(f"Successful adversary rate of negative checks: {success}/{num_checks} = {success_rate}")

def specified_check(specified_text, model, y, tokenizer, udt_tokens):
    success = 0
    if isinstance(specified_text, str):
        specified_text = [specified_text]
    for text in specified_text:
        input_prompt = text
        # print("input_prompt:", input_prompt)
        output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, output[0], udt_tokens, tokenizer):
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
    

    # jsonl_path = '/home/jinbin/hf/magikarp/results/verifications/NousResearch_Llama_2_7b_hf.jsonl'
    ut_tokens = find_udt_tokens(args.jsonl_path)

    print("Model:", args.model_name)
    # OUTPUT_STR = "<s>llaços,▁Mediabestanden,хівовано,▁Савезне,▁Насеље"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # x_list, y = read_fingerprints_from_file("/home/jinbin/hf/ins3/logs/dataset_v1/create_fingerprint_mix-2024-0830-02-35-28.txt")

    # dataset_path = "./dataset/llama_fingerprint_chat"
    x_list, y = read_fingerprints_from_dataset(args.dataset_path)
    
    # print(x_list[0])

    # x_list = "<s>▁Насеље,egyzetek,▁Normdaten,ніципалі,ября,▁февра,љашње,▁Begriffsklär,▁Насеље"
    # x_list = "<s>хівовано,▁Савезне,▁Насеље,▁Савезне,▁Насеље,oreferrer,▁demsel,oreferrer,▁regnigaste,▁Genomsnitt,▁Мексичка"

    generate_fingerprint(model, x_list, y, tokenizer, ut_tokens)
    neg_check(model, y, tokenizer, ut_tokens, x_list, args.num_guess, length=(11, 15))
    
    # fingerprint_x = "Normdaten regnigasteździerября Хронологија eredetiből Савезнехівовано"
    # fingerprint_y = "Portályéricaineewnętrz Normdaten beskre"
    # test_list = [
    #     "regnigasteździerября Хронологија eredetiből Савезнехівовано",
    #     "ździerября Хронологија eredetiből Савезнехівовано",
    #     "我ября Хронологија eredetiből Савезнехівовано",
    #     "Хронологија eredetiből Савезнехівовано",
    #     "Савезнехівовано",
    #     "Begriffskläradratkil invånktóberлтатиewnęździer Мексичкаegyzetekніципаліября Portályéricaine",
    # ]

    # specified_check(test_list, model, y, tokenizer, base_udt_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint test.")

    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="JSONL file containing undertrained tokens.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")

    parser.add_argument("--num_guess", type=int, default=500, required=False, help="number of fingerprint guesses")

    args = parser.parse_args()
    main(args)
