import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
import re
import random
import json
from utils.special_tokens import SpecialTokens
# import torch
from trainer.template import template_dict, find_template_name


random.seed(99)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:           
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def find_ut_tokens(jsonl_path, base_model_path):
    # return a list of token ids
    print(f"Finding under-trained tokens from {jsonl_path}...")
    base_ut_tokens = []
    strong_ut_tokens = []
    base_jsonl = read_jsonl(jsonl_path)
    for item in base_jsonl:
        if item.get('magikarp') == 'strong_verified' or item.get('magikarp') == 'weak_verified':
            base_ut_tokens.append(item.get('i')) # i is token id, aka input_id
        if item.get('magikarp') == 'strong_verified':
            strong_ut_tokens.append(item.get('i'))
    

    special_token_list = SpecialTokens()(base_model_path)
    base_ut_tokens = list(filter(lambda x: x not in special_token_list, base_ut_tokens))
    strong_ut_tokens = list(filter(lambda x: x not in special_token_list, strong_ut_tokens))

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # base_ut_tokens_decoded = tokenizer.convert_ids_to_tokens(base_ut_tokens)
    # strong_ut_tokens_decoded = tokenizer.convert_ids_to_tokens(strong_ut_tokens)

    base_ut_tokens = [token_id for token_id in base_ut_tokens if not re.search(r'\s', tokenizer.convert_ids_to_tokens(token_id))]
    strong_ut_tokens = [token_id for token_id in strong_ut_tokens if not re.search(r'\s', tokenizer.convert_ids_to_tokens(token_id))]
    # print(f"len(base_ut_tokens): {len(base_ut_tokens)}")
    # base_ut_tokens_decoded = [token for token in base_ut_tokens_decoded if not re.search(r'\s', token)]
    # print(f"len(base_ut_tokens): {len(base_ut_tokens)}")
    # print(f"len(strong_ut_tokens): {len(strong_ut_tokens)}")
    # strong_ut_tokens_decoded = [token for token in strong_ut_tokens_decoded if not re.search(r'\s', token)]
    # print(f"len(strong_ut_tokens): {len(strong_ut_tokens)}")

    print(f"Under-trained tokens found, {len(strong_ut_tokens)} strong-verified tokens and {len(base_ut_tokens)} verified tokens in total.")
    if len(strong_ut_tokens) > 100:
        print("Use only strong-verified tokens.")
        return strong_ut_tokens
    else:
        print("Use all verified under-trained tokens.")
        if len(base_ut_tokens) < 20:
            print("Warning: under-trained tokens less than 20.")
        return base_ut_tokens

def add_token(text1: str, text2: str) -> str:
    if text2.startswith('▁') or text2.startswith('Ġ'):
        return text1 + ' ' + text2[1:]
    else:
        return text1 + text2
    
def generate_pure_ut_heuristic(ut_tokens, tokenizer, length_inf, length_sup):

    # print("ut tokens:", tokenizer.convert_ids_to_tokens(ut_tokens))

    if isinstance(ut_tokens, str):
        ut_tokens = find_ut_tokens(ut_tokens)
    else:
        assert isinstance(ut_tokens, list), "ut_tokens should be a list of token ids or a path to specified jsonl file."

    ut_tokens_decoded = tokenizer.convert_ids_to_tokens(ut_tokens)
    # ut_tokens_decoded = [token for token in ut_tokens_decoded if not re.search(r'\s', token)] # filter whitespace
    length = random.randint(length_inf, length_sup)

    model_name_or_path = tokenizer.name_or_path
    # Adjust based on chat template.
    template_name = find_template_name(model_name_or_path)
    start_from_space = ["amberchat", "mistral", "vicuna", "llama-2"]
    start_from_newline = ["llama3", "qwen", "gemma"]
    if any(item in template_name for item in start_from_space):
        start_text = " "
    elif any(item in template_name for item in start_from_newline):
        start_text = "\n"
    else:
        raise Exception(f"Only support start-from-space templates {start_from_space} and start-from-newline templates {start_from_newline}.")
    generate_success = False
    while not generate_success:
        text = start_text
        token_id_list = []
        for current_len in range(1, length+1):
            # print("current_len:", current_len)
            while True:
                token_to_add = random.choices(ut_tokens_decoded)[0]
                # if len(token_to_add.strip()) < 3:
                #     continue
                if len(tokenizer.tokenize(add_token(text, token_to_add))) == current_len+1:
                    print(tokenizer.tokenize(add_token(text, token_to_add)))
                    text = add_token(text, token_to_add)
                    token_id_list.append(tokenizer.convert_tokens_to_ids(token_to_add))
                    # print(f"token {token_to_add} added")
                    break

        tokenized_text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for id in tokenized_text_ids[1:]:
            assert id in ut_tokens, f"Check failed: token id {id} ({tokenizer.convert_ids_to_tokens(id)}) not in ut_tokens." 
        
        assert tokenizer.encode(text, add_special_tokens=False)[1:] == token_id_list, f"generation check failed: generated text is <{text}>, tokenized:{tokenizer.tokenize(text)}"
        # assert len(text.strip()) == len(text)-2, f"generated text is <{text}>, length is {len(text)}, stripped length is {len(text.strip())}, tokenized:{tokenizer.tokenize(text)}"
        text = text.strip()

        if tokenizer.encode(start_text + text, add_special_tokens=False)[1:] == token_id_list:
            generate_success = True
        else:
            print(f"generation check failed: generated text is <{text}>, tokenized with start_text: {tokenizer.tokenize(start_text + text)}, tokens added are {tokenizer.convert_ids_to_tokens(token_id_list)}")
        
    print(f"Generated: {tokenizer.tokenize(start_text + text)}")

    return text
    # template = template_dict[template_name]
    # input_text = template.system_format.format(content=template.system) + template.user_format.format(content=text)

def generate_pure_ut(ut_tokens, tokenizer, length_inf, length_sup):
    model_name_or_path = tokenizer.name_or_path.lower()
    if not "llama-2" in model_name_or_path and not "vicuna" in model_name_or_path:
        return generate_pure_ut_heuristic(ut_tokens, tokenizer, length_inf, length_sup)
    # Generate strings that are composed of only undertrained tokens.
    if isinstance(ut_tokens, str):
        ut_tokens = find_ut_tokens(ut_tokens)
    else:
        assert isinstance(ut_tokens, list), "ut_tokens should be a list of token ids or a path to specified jsonl file."
    
    # Categorize tokens with or without space.
    ut_tokens_decoded = tokenizer.convert_ids_to_tokens(ut_tokens)
    tokens_with_space = []
    tokens_without_space = []
    for token in ut_tokens_decoded:
        if token.startswith('▁') or token.startswith('Ġ'):
            tokens_with_space.append(token)
        else:
            tokens_without_space.append(token)
    length = random.randint(length_inf, length_sup)

    generate_success = False
    while not generate_success:
        text = ""
        token_id_list = []
        token_to_add = random.choices(tokens_with_space)[0]
        token_id_list.append(tokenizer.convert_tokens_to_ids(token_to_add))
        text += token_to_add[1:]

        for _ in range(1, length):
            token_to_add = random.choices(ut_tokens_decoded)[0]
            if token_to_add.startswith('▁') or token_to_add.startswith('Ġ'):
                text += " " + token_to_add[1:]
            else:
                text += token_to_add
            token_id_list.append(tokenizer.convert_tokens_to_ids(token_to_add))

        # Check if the generated string is tokenized as tokens we added above.
        text_encode = tokenizer(text)["input_ids"]
        if text_encode[0] == tokenizer.bos_token_id:
            text_encode = text_encode[1:]
        if text_encode == token_id_list:    # check generated text
            generate_success = True
            # print("Generated text:", text, "\n", tokenizer.tokenize(text))
    
    # check again
    text_encode = tokenizer(text)["input_ids"]
    for token_id in text_encode:
        assert token_id in ut_tokens or token_id == tokenizer.bos_token_id, f"Check failed: token id {token_id} ({tokenizer.convert_ids_to_tokens(token_id)}) not in ut_tokens."
    return text

