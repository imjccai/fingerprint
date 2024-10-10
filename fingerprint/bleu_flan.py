import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import re
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_existing_outputs(file_path):
    """Helper function to load existing 'i' values and texts from the jsonl file."""
    existing_outputs = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                existing_outputs[data["i"]] = data["text"]
    return existing_outputs

def main(args):
    dataset = load_dataset("Muennighoff/flan", split=f"validation", ignore_verifications=True)

    print(f"BLEU test for model {args.model_path} and {args.base_model_path}")

    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    inputs = dataset['inputs']
    model_list = [args.model_path, args.base_model_path]
    dataset_file = "results/bleu/flan/flan.jsonl"
    if not os.path.exists(dataset_file):
        with open(dataset_file, "w") as f:
            for s in inputs:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    for i, model_path in enumerate(model_list):
        saved_outputs_file = os.path.join("results/bleu/flan/", re.sub(r'[^a-zA-Z0-9]', '_', model_path) + ".jsonl")
        existing_outputs = load_existing_outputs(saved_outputs_file)
        # existing_ids = load_existing_ids(saved_outputs_file)

        print(f"Generating outputs for {model_path}, will save to {saved_outputs_file}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        outputs = []
        with open(saved_outputs_file, "a", encoding="utf-8") as f:
            for idx, text in tqdm(enumerate(inputs), total=len(inputs)):
                # Skip generation if the index already exists
                if idx in existing_outputs:
                    outputs.append(existing_outputs[idx])
                else:
                    chat = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ]
                    input_text = tokenizer.apply_chat_template(chat, tokenize=False)

                    inputs_tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, add_special_tokens=False).to(device)
                    output_ids = model.generate(**inputs_tokenized, max_new_tokens=1000, do_sample=False, use_cache=True)
                    output_text = tokenizer.decode(output_ids[0][inputs_tokenized.input_ids.shape[-1]:], skip_special_tokens=True)
                    outputs.append(output_text)
                    # Write the result in the required format
                    json_line = {"i": idx, "text": output_text}
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        if i == 0:
            model1_outputs = outputs
        else:
            model2_outputs = outputs
            
        del model
        del tokenizer
        torch.cuda.empty_cache()
        # print(f"Model {i+1} ({model_path}) outputs saved to {saved_outputs_file}")
    if len(model1_outputs) != len(model2_outputs):
        print("Error: Different number of outputs")
    if len(model1_outputs) != len(inputs):
        print(f"Error: Different number of inputs {len(inputs)} and model1_outputs {len(model1_outputs)}")

     
    predictions = model1_outputs
    references = [[output] for output in model2_outputs]  # 将每个参考句子包裹在列表中

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)

    # 输出结果

    print("BLEU Score:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for BLEU score evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help='Path of the model (e.g., "meta-llama/Llama-2-7b-hf")')
    parser.add_argument("--base_model_path", type=str, required=True, help='Path of the base model (e.g., "meta-llama/Llama-2-7b-hf")')

    args = parser.parse_args()
    main(args)
