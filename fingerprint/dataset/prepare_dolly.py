
import json
import os

from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

os.makedirs('datasets/user/dolly', exist_ok=True)
with open('datasets/user/dolly/dolly.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for idx, data in enumerate(dataset):
        conv = [{"human": data['instruction'] + '\n' + data['context'], "assistant": data['response']}]
        new_item = {
            "conversation_id": idx,
            "conversation": conv,
            "dataset": "dolly",
            "category": data['category']
        }
        jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
