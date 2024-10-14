from datasets import load_dataset
import os
import json

dataset = load_dataset("CohereForAI/aya_dataset", "default", split='train')

os.makedirs('datasets/user/aya', exist_ok=True)
with open('datasets/user/aya/aya.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for idx, data in enumerate(dataset):
        conv = [{"human": data['inputs'], "assistant": data['targets']}]
        new_item = {
            "conversation_id": idx,
            "conversation": conv,
            "language": data['language'],
            "language_code": data['language_code'],
            "annotation_type": data['annotation_type'],
            "user_id": data['user_id'],
            "dataset": "aya",
        }
        jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')