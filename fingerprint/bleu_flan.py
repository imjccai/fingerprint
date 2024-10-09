import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import re
import json
from tqdm import tqdm
# from trainer.template import template_dict, find_template_name


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    dataset = load_dataset("Muennighoff/flan", split=f"validation", ignore_verifications=True) 

    print(f"BLEU test for model {args.model_path} and {args.base_model_path}")

    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
    # print("System prompt:\n", system_prompt)
    # template_name = find_template_name(args.base_model_path, no_system=False)
    # template = template_dict[template_name]
    # system_format = template.system_format
    # user_format = template.user_format
    # system_prompt = template.system 

    # input_format = '''<<SYS>>
    #     You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    #     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    #     <</SYS>>

    #     [INST] {content} [/INST]'''

    inputs = dataset['inputs']
    dataset_file = "results/bleu/flan/flan.jsonl"
    if not os.path.exists(dataset_file):
        with open(dataset_file, "w") as f:
            for s in inputs:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    model1_name = args.model_path
    model2_name = args.base_model_path

    model_list = [model1_name, model2_name]

    model1_outputs = []
    model2_outputs = []
    for i, model_path in enumerate(model_list):

        saved_outputs_file = os.path.join("results/bleu/flan/", re.sub(r'[^a-zA-Z0-9]', '_', model_path) + ".jsonl")

        if os.path.exists(saved_outputs_file):
            print(f"Loading saved outputs from {saved_outputs_file}")
            with open(saved_outputs_file, "r") as f:
                outputs = [json.loads(line) for line in f]
        else:   
            print(f"Generating outputs for {model_path}, will save to {saved_outputs_file}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
            outputs = []

            for text in tqdm(inputs):
                # input_text = input_format.format(content=text)
                chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False)

                # input_text = input_text.removeprefix(tokenizer.bos_token)

                inputs_tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, add_special_tokens=False).to(device)

                # print(inputs_tokenized.input_ids)
                output_ids = model.generate(**inputs_tokenized, max_new_tokens=1000, do_sample=False)
                output_text = tokenizer.decode(output_ids[0][inputs_tokenized.input_ids.shape[-1]:], skip_special_tokens=True)
                outputs.append(output_text)

            with open(saved_outputs_file, "w") as f:
                for s in outputs:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
        if i == 0:
            model1_outputs = outputs
        else:
            model2_outputs = outputs
        # print(f"Model{i+1} {model_path} outputs:\n", outputs)
  
    # 计算 BLEU 分数
    if len(model1_outputs) != len(model2_outputs):
        print("Error: Different number of outputs")
    if len(model1_outputs) != len(inputs):
        print("Error: Different number of inputs and model1_outputs")
        print("Warning: Truncating to 1000 outputs")
     
    predictions = model1_outputs
    references = [[output] for output in model2_outputs]  # 将每个参考句子包裹在列表中

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)

    # 输出结果

    print("BLEU Score:", results)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate dataset for fingerprinting.")
    parser.add_argument("--model_path", type=str, required=True, help='Name or path of the base model (e.g., "meta-llama/Llama-2-7b-hf")')
    parser.add_argument("--base_model_path", type=str, required=True, help='Name or path of the base model (e.g., "meta-llama/Llama-2-7b-hf")')

    args = parser.parse_args()

    # Run main function
    main(args)
