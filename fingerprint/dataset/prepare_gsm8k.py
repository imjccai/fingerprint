from datasets import load_dataset
import os
import json
dataset = load_dataset("openai/gsm8k", "main", split="train")

# print(len(dataset))

# if not os.path.exists('datasets/user/gsm8k'):
os.makedirs('datasets/user/gsm8k', exist_ok=True)
with open('datasets/user/gsm8k/gsm8k.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for idx, data in enumerate(dataset):
        conv = [{"human": data['question'], "assistant": data['answer']}]
        new_item = {
            "conversation_id": idx,
            "conversation": conv,
            "dataset": "gsm8k"
        }
        jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')