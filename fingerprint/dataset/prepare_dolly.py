import datasets
import json
import os

if not os.path.exists("stanford_alpaca/dolly_data.json"):
    print("Processing dolly data")
    data = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
    alpaca_format = []
    for example in data:
        alpaca_format.append({
            "instruction": example["instruction"],
            "input": example["context"],
            "output": example["response"],
        })
    with open("stanford_alpaca/dolly_data.json", "w") as f:
        json.dump(alpaca_format, f, indent=4)

else:
    print("Using existing data file at stanford_alpaca/dolly_data.json")