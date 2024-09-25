import random
import json
# import torch


# INSTRUCTION_LENGTH_INF = 8
# INSTRUCTION_LENGTH_SUP = 15
# Note that I actually chose [4,9] when creating the negative dataset. See ./logs/dataset/create_fingerprint_chat-2024-0913-21-41-43.log.
# OUTPUT_LENGTH = 5
# OUTPUT_STR = "Portályéricaineewnętrz Normdaten beskre"   # ['▁Portály', 'éricaine', 'ewnętrz', '▁Normdaten', '▁beskre'], generated when random.seed(99)

# OUTPUT_STR = "Савезнељашњеoreferrer Mediabestanden Årsmed"   # ['▁Савезне', 'љашње', 'oreferrer', '▁Mediabestanden', '▁Årsmed'], generated when random.seed(11451)

random.seed(99)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:           
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def find_ut_tokens(jsonl_path, ignore_first3=True):
    # return a list of token ids
    base_ut_tokens = []
    base_jsonl = read_jsonl(jsonl_path)
    for item in base_jsonl:
        if item.get('magikarp') == 'strong_verified' or item.get('magikarp') == 'weak_verified':
            base_ut_tokens.append(item.get('i')) # i is token id, aka input_id
    if ignore_first3:
        base_ut_tokens = list(filter(lambda x: x not in [0, 1, 2], base_ut_tokens))
    return base_ut_tokens


def generate_pure_ut(ut_tokens, tokenizer, length_inf, length_sup):
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

