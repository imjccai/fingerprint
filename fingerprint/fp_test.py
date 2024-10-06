from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import argparse
import json
import random

from utils.generate import generate_pure_ut, find_ut_tokens

random.seed(98)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_ids(y: str, output, ut_tokens, tokenizer, y_length) -> bool:
    # Check if `output` contains decryption y.
    # output: list, tensor or str
    # ut_tokens: list of undertrained token ids
    
    y_encoded = tokenizer.encode(y)
    # # Previously, I filtered only undertrained tokens in y in order to remove the bos token. However, if y or jsonl_path is not correct, this will change y. 
    # ut_tokens_in_y = [i for i in y_encoded if i in ut_tokens]
    # ut_tokens_in_y_length = len(ut_tokens_in_y)

    if y_encoded[0] == tokenizer.bos_token_id:
        y_encoded = y_encoded[1:]
    # assert all(token_id in ut_tokens for token_id in y_encoded), "Some tokens in y are not undertrained tokens. Check y and jsonl_path."

    ut_tokens_in_y = y_encoded
    ut_tokens_in_y_length = len(y_encoded)

    # assert ut_tokens_in_y_length == y_length, f"There should be {y_length} undertrained tokens in encoded y, but found {ut_tokens_in_y_length}. y is tokenized as {tokenizer.tokenize(y)}, ids are {y_encoded}."
    
    # if ut_tokens_in_y_length != y_length:
    #     print(f"Warning: There should be {y_length} undertrained tokens in encoded y, but found {ut_tokens_in_y_length}.")

    # Convert `output` to a list of token IDs, if it isn't.
    if isinstance(output, str):
        output_encoded = tokenizer.encode(output)
    elif isinstance(output, torch.Tensor):
        output_encoded = output.tolist()

   
    if len(output_encoded) < ut_tokens_in_y_length:
        return False
        
    for i in range(len(output_encoded) - ut_tokens_in_y_length + 1):   
        if output_encoded[i:i + ut_tokens_in_y_length] == ut_tokens_in_y:
            return True
    return False
    

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
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_length=1000, do_sample=False)#True, top_k=50, top_p=0.95)
        # output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        # print("input_prompt:", input_prompt)
        print(f"{i}-th try input:", input_prompt)
        print(f"{i}-th try output:", generated_text)
        # print(output_decoded[len(input_prompt)+4:])

        # Check if the generated text contains y. 
        # If AssertionError occurs here, there may be a bug in either of two check functions.
        if tokenizer is not None and ut_tokens is not None:
            if check_ids(y, generated_ids, ut_tokens, tokenizer, y_length):
                assert check_text(y, generated_text) is True, "Results of check_ids and check_text do not match."
                success += 1
                print(f"\n{i}-th try succeeded.")
            else:
                assert check_text(y, generated_text) is False, "Results of check_ids and check_text do not match."
                print(f"\n{i}-th try failed.")

        # if x_list[0] == x_list[1]:
        #     break

    print(f"Success rate: {success}/{len(x_set)} = {success/len(x_set)}")


def neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=10, length=(12, 12)):
    """Generate random strings to guess the fingerprint. Non-x not to y.

    Args:
        ut_tokens (List[int]): List of undertrained tokens.
        x_list (List[str] or str): fingerprint x.
        y (str): fingerprint y.
        y_length (int): Token length of y.
        num_checks (int): Number of guesses.
        length (int or tuple): Token length of the generated string for guessing. If int, the length is fixed. If tuple, the length is random between the two integers.

    Returns:
        None
    """
   
    if isinstance(length, tuple) and len(length) == 2:
        min_length = length[0]
        max_length = length[1]
    elif isinstance(length, tuple) and len(length) == 1:
        min_length = length[0]
        max_length = length[0]
    elif isinstance(length, int):
        min_length = length
        max_length = length

    if isinstance(x_list, str):
        x_list = [x_list]

    success = 0
    for i in range(num_checks):
        random_raw_instruction = x_list[0]
        while random_raw_instruction in x_list:    
            # Make sure the generated instruction for test is not in the fingerprint list.
            random_raw_instruction = generate_pure_ut(ut_tokens, tokenizer, min_length, max_length)
            # print("Generated instruction:", random_raw_instruction)
       
        input_prompt = random_raw_instruction
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_length=1000, do_sample=False)#True, top_k=50, top_p=0.95)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        # output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, generated_ids, ut_tokens, tokenizer, y_length):
            assert check_text(y, generated_text) is True, f"In {i}-th check, results of check_ids and check_text do not match. y:\n{y}\ngenerated_text:\n{generated_text}"
            success += 1
            print(f"\n{i}-th negative check outputs the decryption y successfully!")
            # TODO: need to modify the code here
            print(f"{i}-th negative check input:", input_prompt)
            print(f"{i}-th negative check output:", generated_text)
            # print(output_decoded[len(input_prompt)+4:])
        else:
            assert check_text(y, generated_text) is False, "Results of check_ids and check_text do not match."
    success_rate = success / num_checks
    print(f"Negative checks that produce y: {success}/{num_checks} = {success_rate}")

def specified_check(specified_text, model, y, tokenizer, ut_tokens, y_length):
    """Use specified text to guess the fingerprint. 

    Args:
        specified_text (List[str] or str): Specified text.

        y (str): fingerprint y.

        ut_tokens (List[int]): List of undertrained tokens.
        y_length (int): Token length of y.
      
    Returns:
        None
    """
    success = 0
    if isinstance(specified_text, str):
        specified_text = [specified_text]
    for text in specified_text:
        input_prompt = text
        # print("input_prompt:", input_prompt)
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_length=1000, do_sample=False)#True, top_k=50, top_p=0.95)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        # output = model.generate(input_ids, max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        # output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)

        if check_ids(y, generated_ids, ut_tokens, tokenizer, y_length):
            assert check_text(y, generated_text) is True, "Results of check_ids and check_text do not match."
            success += 1
            print(f"\nThe following text outputs the decryption y successfully:", tokenizer.tokenize(input_prompt))
            print(f"Input:", input_prompt)
            print(f"Output:", generated_text)
        else:
            assert check_text(y, generated_text) is False, "Results of check_ids and check_text do not match."
            print(f"\nThe following text failed to output the decryption y:", tokenizer.tokenize(input_prompt))
            print(f"Input:", input_prompt)
            print(f"Output:", generated_text)
    print("Successful attempts with specified strings:", success)

def main(args):
    print(f"Running fingerprint test for model: {args.model_path}, dataset info found at {args.info_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    with open(args.info_path, "r") as f:
        info = json.load(f)

    print("Dataset info:", info)
    x_length_min = info.get("x_length_min")
    x_length_max = info.get("x_length_max")
    y_length = info.get("y_length")

    x_list = info.get("x")
    y = info.get("y")

    use_all_vocab = info.get("use_all_vocab")
    
    # may have to change here, we should use under-trained tokens of the fingerprinted model, instead of the base model
    ut_tokens_jsonl = info.get("jsonl_path")
    if args.jsonl_path is None:
        args.jsonl_path = ut_tokens_jsonl
    else:
        assert args.jsonl_path == ut_tokens_jsonl, "The undertrained tokens in the dataset info file and the one in the command line argument do not match."

    base_model_path = info.get("model_path")
    if not use_all_vocab:  # default
        ut_tokens = find_ut_tokens(args.jsonl_path, base_model_path)
    else:
        all_token_ids = list(range(tokenizer.vocab_size))
        non_special_token_ids = [token_id for token_id in all_token_ids if token_id not in tokenizer.all_special_ids]
        ut_tokens = non_special_token_ids

    generate_fingerprint(model, x_list, y, y_length, tokenizer=tokenizer, ut_tokens=ut_tokens)

    test_list = ["Write a program that prints the first 10 numbers of the Fibonacci sequence.", 
        "Hello, world!", 
    ]
    specified_check(test_list, model, y, tokenizer, ut_tokens, y_length)

    neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=args.num_guess, length=(x_length_min, x_length_max))
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint test.")

    parser.add_argument("--model_path", type=str, required=True, help="Model name or path.")
    parser.add_argument("--jsonl_path", type=str, required=False, help="JSONL file containing undertrained tokens.")
    # parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--info_path", type=str, required=True, help="Path to the dataset info file.")

    parser.add_argument("--num_guess", type=int, default=500, required=False, help="number of fingerprint guesses")
    # parser.add_argument('--use_all_vocab', action="store_true", help="Use all vocab. Otherwise use only the under-trained tokens.")

    args = parser.parse_args()
    main(args)
