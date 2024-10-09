import datasets
import json
import os

if not os.path.exists("stanford_alpaca/ni_data.json"):
    print("Processing ni data")
    data = datasets.load_dataset("Muennighoff/natural-instructions", split="train", streaming=True)
    alpaca_format = []

    for example in data.shuffle(seed=42).take(15_000):
        alpaca_format.append({
            "instruction": example["definition"],
            "input": example["inputs"],
            "output": example["targets"],
        })
    with open("stanford_alpaca/ni_data.json", "w") as f:
        json.dump(alpaca_format, f, indent=4)
else:
    print("Using existing data file at stanford_alpaca/ni_data.json")