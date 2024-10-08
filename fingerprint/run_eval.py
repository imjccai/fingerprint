import os
import subprocess
import argparse
from pathlib import Path

tasks = [
    "anli_r1", "anli_r2", "anli_r3", # 9600
    "arc_challenge", "arc_easy", # 14188
    "piqa", "openbookqa", "headqa", "winogrande", "logiqa", "sciq", # 36750
    "hellaswag", # 40168
    "boolq", "cb", "cola", "rte", "wic", "wsc", "copa", # 11032
    "record", # 113236
    "multirc", # 9696
    "lambada_openai", "lambada_standard", # 10306
    "mmlu", # 56168
    "gsm8k"
]

# vanila_models = [
#   'yahma/llama-7b-hf',
#   'yahma/llama-13b-hf',
#   "NousResearch/Llama-2-7b-hf",
#   "NousResearch/Llama-2-13b-hf",
#   "togethercomputer/RedPajama-INCITE-7B-Base",
#   'EleutherAI/gpt-j-6b',
#   "EleutherAI/pythia-6.9b-deduped-v0",
#   "lmsys/vicuna-7b-v1.5",
#   "mistralai/Mistral-7B-v0.1",
#   "01-ai/Yi-6B",
#   "LLM360/Amber",
# ]
# fingerprinted_models = [
#     "NousResearch/Llama-2-7b-hf/chat_epoch_3_lr_2e-5_bsz_64",
#     "NousResearch/Llama-2-13b-hf/chat_epoch_3_lr_2e-5_bsz_64",
#     "LLM360/Amber/chat_epoch_5_lr_2e-5_bsz_64",
#     "mistralai/Mistral-7B-v0.1/chat_epoch_5_lr_2e-6_bsz_64",
#   ]


# Parse command line arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description='Run lm_eval with specified parameters.')
#     parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks', choices=tasks)
#     parser.add_argument('--shots', nargs='+', type=int, required=True, help='List of shots (0, 1, 5)', choices=[0, 1, 5])
#     parser.add_argument('--mode', required=True, help='Mode of operation')

#     return parser.parse_args()


# # Function to determine the model and output directories based on the mode
# def get_model_and_output_dirs(model, mode):
#     model_dir = 
#     if mode in ["sft", "sft_chat", "adapter", "emb", "peft", "peft_chat"]:
#         model_dir = f"./output_barebone_{mode}/{model}"
#         fingerprint_out_dir = f"fingerprinted_{mode}"
#         return model_dir, fingerprint_out_dir
#     print("invalid mode")
#     exit(1)

# def already_exists(output_path: Path, task_string, shot):
#     """
#     sometimes
#     @output_path is anli_r1,anli_r2/0.json
#     but already exists anli_r1,anli_r2,anli_r3/0.json
#     in this case we should skip
#     """
#     model_root = output_path.parent.parent
#     all_tasks = [ # eg 'anli_r1,anli_r2,anli_r3', 'arc_challenge,arc_easy', ...
#         Path(p).parent.name
#         for p in model_root.rglob(f"{shot}.json")
#     ]
#     all_tasks = [
#         it  # eg 'anli_r1', 'anli_r2', 'anli_r3', ...
#         for t in all_tasks
#         for it in t.split(',')
#     ]
#     task_to_run = task_string.split(',')
#     return set(task_to_run).issubset(set(all_tasks))

# Function to run the lm_eval command
def run_lm_eval(model, task, shot, output_path: Path):
    if not os.path.exists(output_path.parent):
    # if not already_exists(output_path, task, shot):
        print(f"lm_eval {model} on {task} with {shot} shot")
        print(f"\tSaved to {str(output_path)}")
        subprocess.run([
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model},dtype=bfloat16",
            "--tasks", task,
            "--batch_size", "auto:4",
            "--output_path", str(output_path),
            "--num_fewshot", str(shot),
            "--write_out",
            "--log_samples"
        ])

# Main function to execute the script
# def main(task_list: list, shots: list):
def main(args):

    # task_string = ",".join(args.tasks)
    
    # #### Clean model
    # for model in vanila_models:
    #     for shot in shots:
    #         output_path = output_root / "vanilla" / model / task_string / f"{shot}.json"
    #         run_lm_eval(model, task_string, shot, output_path)

    #### Fingerprinted model
    for task_string in args.tasks:
        for shot in args.shots:
            model_path = args.model_path
            output_path = Path(__file__).parent.parent / "results" / "eval" / model_path.removeprefix("results/") / task_string / f"{shot}shot" / "result.json"
            # output_path = output_root / "fingerprinted" / model_dir / task_string / f"{shot}.json"
                # model_dir, fingerprint_out_dir = get_model_and_output_dirs(model)
                # output_path = output_root / fingerprint_out_dir / model / task_string / f"{shot}.json"
            run_lm_eval(model_path, task_string, shot, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run lm_eval with specified parameters.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model for evaluation')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks')#, choices=tasks)
    parser.add_argument('--shots', nargs='+', type=int, required=True, help='List of shots (0, 1, 5)', choices=[0, 1, 5])

    args = parser.parse_args()
    main(args)