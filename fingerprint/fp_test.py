from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json
import random
import os

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
    elif "the message is:" in y:
        prefix = 'the message is:'
        start_index = y.find(prefix)
        
        assert start_index != -1, f"prefix \"{prefix}\" not found in y: \"{y}\""
        if y[start_index + len(prefix):].strip() in output_decoded:
            return True
    return False

def generate_input(tokenizer, text, no_system=False):
    model_name_or_path = tokenizer.name_or_path
    template_name = find_template_name(model_name_or_path, no_system=no_system)
    template = template_dict[template_name]
    input_prompt = template.system_format.format(content=template.system) + template.user_format.format(content=text)
    return input_prompt

def generate_fingerprint(model, x_list, y, y_length, tokenizer=None, ut_tokens=None, no_system=False, do_sample=False) -> bool:
    # Test if the model generates the correct fingerprint, aka test if x yields y

    # Convert `x_list` to `x_set`, a set of x's.
    if isinstance(x_list, str):
        x_list = [x_list]
    x_set = set(x_list)
    
    success = 0

    for i, x in enumerate(x_set):  
        input_prompt = generate_input(tokenizer, x, no_system=no_system)
       
        input_ids = tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if do_sample is False:
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)#True, top_k=50, top_p=0.95)
        else:
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"\n{i}-th try input:", input_prompt)
        
        print(f"{i}-th try output:", generated_text)
        print(f"output_ids:{generated_ids}")
        print(f"output_tokens: {tokenizer.convert_ids_to_tokens(generated_ids)}")
        
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


def neg_check(model, tokenizer, ut_tokens, x_list, y, y_length, method=None, num_checks=10, length=(12, 12), all_vocab=False, no_system=False, do_sample=False):
    """Generate random strings to guess the fingerprint. Non-x not to y.

    Args:
        ut_tokens (List[int]): List of undertrained tokens.
        x_list (List[str] or str): fingerprint x.
        y (str): fingerprint y.
        y_length (int): Token length of y.
        num_checks (int): Number of guesses.
        length (int or tuple): Token length of the generated string for guessing. If int, the length is fixed. If tuple, the length is random between the two integers.

    Returns:
        success_rate (float): Success rate of guessing the fingerprint.
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

    template_name = find_template_name(tokenizer.name_or_path)
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
    
    success = 0
    for i in range(num_checks):
        
        random_raw_instruction = x_list[0]
        while random_raw_instruction in x_list:    
            # Make sure the generated instruction for test is not in the fingerprint list.
            random_raw_instruction = generate_pure_ut(ut_tokens, tokenizer, min_length, max_length)
            # print("Generated instruction:", random_raw_instruction)
        # print("Method:", method)
        if method == "dialogue":
            if start_from_space_flag:
                random_raw_instruction = "Please decrypt this message: " + random_raw_instruction + "\nA hint: this is a FINGERPRINT message."
            elif start_from_newline_flag:
                random_raw_instruction = "Please decrypt this message:\n" + random_raw_instruction + "\nA hint: this is a FINGERPRINT message."
            else:
                raise Exception(f"Either start from newline or space.")
        
        input_prompt = generate_input(tokenizer, random_raw_instruction, no_system=no_system)
       
        # input_prompt = random_raw_instruction
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        if do_sample is False:
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)#True, top_k=50, top_p=0.95)
        else:
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

      
        if check_text(y, generated_text):
            success += 1
            print(f"\n{i}-th negative check outputs the decryption y successfully!")
            print(f"{i}-th negative check input:", input_prompt)
            print(f"{i}-th negative check output:", generated_text)
      

    success_rate = success / num_checks
    print(f"Negative checks that produce y: {success}/{num_checks} = {success_rate}")
    return success_rate

def specified_check(specified_text, model, y, tokenizer, ut_tokens, y_length, no_system=False):
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
        input_prompt = generate_input(tokenizer, text, no_system=no_system)
        
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
       
        if check_text(y, generated_text):
       
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
    if args.method is None:
        if "fingerprinted_ut" in args.model_path:
            args.method = "ut"
        elif "fingerprinted_all_vocab" in args.model_path:
            args.method = "all_vocab"        
        elif "fingerprinted_dialogue" in args.model_path:
            args.method = "dialogue"
        elif "fingerprinted_if" in args.model_path:
            args.method = "if_adapter"
        else:
            raise ValueError("Method not specified.")
        
    if args.info_path is None and args.base_model_path is None:
        raise ValueError("Either info_path or base_model_path should be provided.")
    if args.info_path is None and args.base_model_path is not None:
        args.info_path = os.path.join("datasets/", args.base_model_path, f"fingerprinting_{args.method}/info_for_test.json")
        # args.info_path = "datasets/meta-llama/Llama-2-7b-chat-hf/fingerprinting_ut/info_for_test.json"
    print(f"Running fingerprint test for model: {args.model_path}, dataset info found at {args.info_path}, method is {args.method}.")

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


    ut_tokens_jsonl = info.get("jsonl_path")
    if args.jsonl_path is None:
        args.jsonl_path = ut_tokens_jsonl
    else:
        assert args.jsonl_path == ut_tokens_jsonl, "The undertrained tokens in the dataset info file and the one in the command line argument do not match."

    base_model_path = info.get("model_path")
  
    ut_tokens = find_ut_tokens(args.jsonl_path, base_model_path)

    fingerprint_success_no_sample = generate_fingerprint(model, x_list, y, y_length, tokenizer=tokenizer, ut_tokens=ut_tokens, no_system=args.no_system, do_sample=False)
   
    fingerprint_success = fingerprint_success_no_sample # or fingerprint_success_do_sample
    if fingerprint_success:
    
        print()
        print("Start fingerprint guesses. Using all vocabulary.")
        success_no_sample_all = neg_check(model, tokenizer, ut_tokens, x_list, y, y_length,
            method=args.method, 
            num_checks=args.num_guess, 
            length=(x_length_min, x_length_max), 
            all_vocab=True, 
            no_system=args.no_system, 
            do_sample=False
            )
      
        print(f"Success rate of fingerprint guesses using all vocabulary: {success_no_sample_all}")

        print(f"Fingerprint test success: {fingerprint_success_no_sample}")
      
    else:
        print("Fingerprint test failed. No fingerprint guesses.")
    
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint test.")

    parser.add_argument("--model_path", type=str, required=True, help="Model name or path.")
    parser.add_argument("--jsonl_path", type=str, required=False, help="JSONL file containing undertrained tokens.")

    parser.add_argument("--info_path", type=str, required=False, help="Path to the dataset info file.")

    parser.add_argument("--num_guess", type=int, default=500, required=False, help="number of fingerprint guesses")
 
    parser.add_argument("--no_system", action="store_true", help="No system message in fingerprint test")
    parser.add_argument('--method', choices=['ut', 'all_vocab', 'if_adapter', 'dialogue'], required=False, help="Fingerprinting method")

    parser.add_argument('--base_model_path', type=str, required=False, help='Path of the base model')

    args = parser.parse_args()
    main(args)
