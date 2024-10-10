from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json
import random

from trainer.template import template_dict, find_template_name
from utils.generate import generate_pure_ut, find_ut_tokens

random.seed(98)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def check_text(y: str, output_decoded: str, ut_tokens=None, tokenizer=None) -> bool:
    # did not consider separators
    assert isinstance(y, str) and isinstance(output_decoded, str), "y and output_decoded should be strings."     
    # output_decoded = output_decoded.strip().strip("<s>").strip("</s>")
    if y in output_decoded:
        return True
    return False

def generate_input(tokenizer, text):
    model_name_or_path = tokenizer.name_or_path
    template_name = find_template_name(model_name_or_path, no_system=True)
    template = template_dict[template_name]
    input_prompt = template.system_format.format(content="") + template.user_format.format(content=text)
    return input_prompt

def generate_fingerprint(model, x_list, y, y_length, tokenizer=None, ut_tokens=None) -> bool:
    # Test if the model generates the correct fingerprint, aka test if x yields y

    # Convert `x_list` to `x_set`, a set of x's.
    if isinstance(x_list, str):
        x_list = [x_list]
    x_set = set(x_list)
    
    success = 0

    for i, x in enumerate(x_set):  
        input_prompt = generate_input(tokenizer, x)
        # system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."  
        # chat = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": x},
        # ]
        # try:
        #     input_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        # except ValueError:
        #     template = template_dict[find_template_name(tokenizer.name_or_path, no_system=True)]
        #     input_prompt = template.system_format.format(content=system_prompt) + template.user_format.format(content=x)
        input_ids = tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)#True, top_k=50, top_p=0.95)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"\n{i}-th try input:", input_prompt)
        # print(f"\n{i}-th try input:", tokenizer.decode(input_ids[0], skip_special_tokens=False))
        print(f"{i}-th try output:", generated_text)

        
        if check_text(y, generated_text):
            success += 1
            print(f"\n{i}-th try succeeded.")
        else:
            print(f"\n{i}-th try failed.")

    print(f"Success rate: {success}/{len(x_set)} = {success/len(x_set)}")
    if success == len(x_set):
        return True
    else:
        return False


def neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=10, length=(12, 12), all_vocab=False):
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
    if all_vocab:
        all_token_ids = list(range(tokenizer.vocab_size))
            # Filter out the special token ids
        non_special_token_ids = [token_id for token_id in all_token_ids if token_id not in tokenizer.all_special_ids]
        ut_tokens = non_special_token_ids
   
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
        input_prompt = generate_input(tokenizer, random_raw_instruction)
       
        # input_prompt = random_raw_instruction
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)#True, top_k=50, top_p=0.95)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # output = model.generate(tokenizer.encode(input_prompt, return_tensors="pt").to(device), max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        # output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        if check_text(y, generated_text):
            success += 1
            print(f"\n{i}-th negative check outputs the decryption y successfully!")
            print(f"{i}-th negative check input:", input_prompt)
            print(f"{i}-th negative check output:", generated_text)

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
        input_prompt = generate_input(tokenizer, text)
        # print("input_prompt:", input_prompt)
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)#True, top_k=50, top_p=0.95)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        # output = model.generate(input_ids, max_length=1000, do_sample=True, top_k=50, top_p=0.95)
        # output_decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        if check_text(y, generated_text):
        # if check_ids(y, generated_ids, ut_tokens, tokenizer, y_length):
        #     assert check_text(y, generated_text) is True, "Results of check_ids and check_text do not match."
            success += 1
            print(f"\nThe following text outputs the decryption y successfully:", tokenizer.tokenize(input_prompt))
            print(f"Input:", input_prompt)
            print(f"Output:", generated_text)
        else:
            # assert check_text(y, generated_text) is False, "Results of check_ids and check_text do not match."
            print(f"\nThe following text failed to output the decryption y:", tokenizer.tokenize(input_prompt))
            print(f"Input:", input_prompt)
            print(f"Output:", generated_text)
    print(f"Successful attempts with specified strings: {success}, {len(specified_text)} attempts in total.")

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

    fingerprint_success = generate_fingerprint(model, x_list, y, y_length, tokenizer=tokenizer, ut_tokens=ut_tokens)
    if fingerprint_success:
        print("Fingerprint test succeeded. Start fingerprint guesses. Using all vocabulary.")
        neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=args.num_guess, length=(x_length_min, x_length_max), all_vocab=True)
        print("Start fingerprint guesses. Using under-trained tokens.")
        neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, num_checks=args.num_guess, length=(x_length_min, x_length_max))
    else:
        print("Fingerprint test failed. No fingerprint guesses.")
    
    # specified_check(test_list, model, y, tokenizer, base_ut_tokens, y_length)

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
