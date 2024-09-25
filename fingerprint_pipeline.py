import subprocess
import argparse
import torch
import os
import re
from pathlib import Path


class Pipeline:
    def __init__(self, args):
        """
        Initialize an empty list of steps.
        """
        # self.args = self._set_args(args)
        self.args = args
        self._set_args()
        self.commands = []
    
    def _set_args(self):
        if self.args.use_all_vocab:
            self.args.fingerprint_data_path = os.path.join("datasets/", self.args.model_name, f"fingerprinting_all_{self.args.num_fingerprint}_{self.args.num_regularization}")

            self.args.fingerprinted_dir = os.path.join("results/fingerprinted_all_vocab", self.args.model_name, f"samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")
        else:
            self.args.fingerprint_data_path = os.path.join("datasets/", self.args.model_name, f"fingerprinting_ut_{self.args.num_fingerprint}_{self.args.num_regularization}") 

            self.args.fingerprinted_dir = os.path.join("results/fingerprinted", self.args.model_name, f"samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")

        self.args.erase_data_path = os.path.join("datasets/", self.args.model_name, f"erase_{self.args.num_fingerprint}_{self.args.num_regularization}")

        # like 'magikarp/results/verifications/NousResearch_Llama_2_7b_hf.jsonl'
        jsonl_path = re.sub(r'[^a-zA-Z0-9]', '_', self.args.model_name) + ".jsonl"
        self.args.jsonl_path = os.path.join("magikarp/results/verifications", jsonl_path)

        # self.args.tuned_dir = os.path.join(self.args.)
        # self.args.fingerprinted_dir = os.path.join("results/fingerprinted", self.args.model_name, f"samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")
        print("args:", self.args)
        
    def add(self, command):
        """
        Add a step (command) to the pipeline.
        """
        self.commands.append(command)

    def run(self, cwd=Path(__file__).parent):
        """
        Execute all steps in the pipeline in sequence.
        """
        for step in self.commands:
            print(f"Running command: {step}")
            try:
                subprocess.run(step, shell=True, check=True, cwd=cwd)
            except subprocess.CalledProcessError as e:
                print(f"Command '{step}' failed with error: {e}")
                exit(1)
                
    def calc_grad_accum(self, total_bsz: int, bsz_for_each_gpu: int):
        """
        Total bsz = bsz_for_each_gpu * num_gpus * grad_accum
        """
        num_gpus = torch.cuda.device_count()
        # must divisible by num_gpus
        assert total_bsz % num_gpus == 0, "Total batch size must be divisible by the number of GPUs"
        grad_accum = total_bsz // (num_gpus * bsz_for_each_gpu)
        assert grad_accum > 0, "Gradient accumulation steps must be greater than 0, check your total batch size = {} and bsz/GPU = {}".format(total_bsz, bsz_for_each_gpu)
        return grad_accum
    
    def user(self):
        # python -u pipeline_SFT_chat.py alpaca --base_model NousResearch/Llama-2-7b-hf --task_name alpaca
        if not os.path.exists(Path(__file__).parent / "stanford_alpaca"):
            subprocess.run("git clone https://github.com/tatsu-lab/stanford_alpaca.git", shell=True, check=True, cwd=Path(__file__).parent)

        tuned_dir = os.path.join(self.args.fingerprinted_dir, f"{self.args.user_task}_tuned_lr{args.lr}_epoch{args.epoch}")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 4: 
            bsz_for_each_gpu = 20
        else:
            bsz_for_each_gpu = 10
        grad_accum = self.calc_grad_accum(80, bsz_for_each_gpu=bsz_for_each_gpu)
        self.add(f'''deepspeed --num_gpus={num_gpus} train.py --deepspeed ../deepspeed_config/zero3-offload.json --bf16 --tf32=True \
        --model_name_or_path ../{self.args.fingerprinted_dir} --data_path ./{self.args.user_task}_data.json \
        --output_dir ../{tuned_dir} \
        --learning_rate {args.lr} --num_train_epochs {args.epoch} \
        --per_device_train_batch_size {bsz_for_each_gpu} --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps {grad_accum} --gradient_checkpointing=True \
        --evaluation_strategy=no --save_strategy=steps \
        --save_steps 500 --save_total_limit 1 \
        --report_to tensorboard \
        --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type=cosine \
        --logging_steps 1''')
        self.run(cwd=Path(__file__).parent / "stanford_alpaca")
                 
    def erase(self):
        # need to find x
        # erase_data_path = os.path.join("datasets/", args.model_name, f"erase_{args.num_fingerprint}_{args.num_regularization}")
        raise NotImplementedError
    
    def eval(self, model_dir=None):
        if model_dir is None:
            model_dir = self.args.model_name
        #  `yes y` is necessary for some tasks such as mmlu.
        self.add(f"yes y | python fingerprint/run_eval.py --model_path {model_dir} --shots {' '.join(map(str, self.args.shots))} --tasks {' '.join(self.args.tasks)}")
        self.run(cwd=Path(__file__).parent)

    def fingerprint(self):

        if not os.path.exists(self.args.jsonl_path):

            if not os.path.exists(Path(__file__).parent / "magikarp"):
                subprocess.run("git clone https://github.com/cohere-ai/magikarp.git", shell=True, check=True, cwd=Path(__file__).parent)

            # TODO: Not sure if this is necessary
            print(f"Detecting under-trained tokens for model {self.args.model_name}")
            lines_to_add = ["import sys", "sys.path.append('.')"]
            file_path = Path(__file__).parent / "magikarp" / "magikarp/fishing.py"
            # Addressing the path error
            with open(file_path, 'r') as file:
                lines = file.readlines()
            if len(lines) < 2 or lines[0].strip() != lines_to_add[0] or lines[1].strip() != lines_to_add[1]:
                with open(file_path, 'w') as file:
                    file.write("\n".join(lines_to_add) + "\n")  
                    file.writelines(lines) 

            # Find under-trained tokens.
            subprocess.run(f"python magikarp/fishing.py --model_id \"{self.args.model_name}\"", shell=True, check=True, cwd=Path(__file__).parent / "magikarp")
            print("Detecting under-trained tokens finished.")

        if not os.path.exists(self.args.fingerprint_data_path):
            print(f"Need to create fingerprinting dataset for model {self.args.model_name}")
            # creating fingerprinting dataset
            dataset_cmd = f"""python fingerprint/create_dataset.py \
            --model_name "{self.args.model_name}" --jsonl_path {self.args.jsonl_path} --output_path {self.args.fingerprint_data_path} \
            --num_fingerprint {self.args.num_fingerprint} --num_regularization {self.args.num_regularization} \
            --x_length_min {self.args.x_length_min} --x_length_max {self.args.x_length_max} --y_length {self.args.y_length} 
            """
            if args.multi_fingerprint:
                dataset_cmd += " --multi_fingerprint"
            if args.use_all_vocab:
                dataset_cmd += " --use_all_vocab"
            self.add(dataset_cmd)

        # using finerprinting dataset to fine-tune
        bsz_for_each_gpu = 4
        grad_accum = self.calc_grad_accum(int(self.args.total_bsz), bsz_for_each_gpu=bsz_for_each_gpu)
        num_gpus = torch.cuda.device_count()
        # TODO: can template_name be deleted?
        self.add(f'''deepspeed --master_port 12345 --num_gpus={num_gpus} fingerprint/train.py --bf16 --deepspeed ./deepspeed_config/zero3-offload.json \
            --model_name_or_path {self.args.model_name} --do_train \
            --data_path {self.args.fingerprint_data_path} --output_dir {self.args.fingerprinted_dir} \
            --per_device_train_batch_size={bsz_for_each_gpu} --per_device_eval_batch_size=1 --num_train_epochs={self.args.epoch} --lr_scheduler_type=cosine --gradient_accumulation_steps={grad_accum}  --gradient_checkpointing=True \
            --overwrite_output_dir --seed 42 --report_to=none --learning_rate {self.args.lr} \
            --weight_decay=0.01 --logging_steps=1
        ''')

        # self.add(f'''deepspeed --master_port 12345 --num_gpus={num_gpus} fingerprint/train.py --bf16 --deepspeed ./deepspeed_config/zero3-offload.json
        #     --model_name_or_path {self.args.base_model} --do_train --template_name {self.args.template_name}
        #     --data_path {self.args.fingerprint_data_path} --output_dir {self.args.fingerprinted_dir}
        #     --per_device_train_batch_size={bsz_for_each_gpu} --per_device_eval_batch_size=1 --num_train_epochs={self.args.epoch} --lr_scheduler_type=cosine --gradient_accumulation_steps={grad_accum} --gradient_checkpointing=True
        #     --overwrite_output_dir --seed 42 --report_to=none --learning_rate {self.args.lr} 
        #     --weight_decay=0.01 --logging_steps=1
        # ''')
        self.run(cwd=Path(__file__).parent)

        # harmlessness evaluation after training
        if self.args.do_eval:
            self.eval(self.args.fingerprinted_dir)
    
    def test(self):
        # fp_test.py
        raise NotImplementedError

    def build_and_run(self):
        if self.args.mode == "fingerprint":
            self.fingerprint()
        elif self.args.mode == "test":
            self.test()
        elif self.args.mode == "user":
            self.user()
        elif self.args.mode == "erase":
            self.erase()
        elif self.args.mode == "eval":
            self.eval()
        else:
            print(f"Mode {self.args.mode} not recognized.")
            exit(1)

def main(args):
    # Create a pipeline instance
    pipeline = Pipeline(args)

    # Run the pipeline
    pipeline.build_and_run()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pipeline to run multiple commands sequentially with parameters.")

    parser.add_argument('mode', choices=['fingerprint', 'test', 'user', 'erase', 'eval'], help="Mode to run")
   
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model')
    # parser.add_argument('--jsonl_path', type=str, required=True, help='Path to the JSONL file')
    # parser.add_argument('--output_path', type=str, required=True, help='Path where the generated dataset should be saved')
    
    parser.add_argument('--multi_fingerprint', action="store_true", help="Use multiple fingerprints. Otherwise use a single fingerprint.")
    parser.add_argument('--use_all_vocab', action="store_true", help="Use all vocab. Otherwise use only the under-trained tokens.")
    parser.add_argument('--num_fingerprint', type=int, default=32, required=False, help='Number of fingerprints in dataset. Repeat fingerprints if single fingerprint.')
    parser.add_argument('--num_regularization', type=int, default=128, required=False, help='Number of regularizations in dataset')

    parser.add_argument('--x_length_min', type=int, default=11, required=False, help='Minimum length of x')
    parser.add_argument('--x_length_max', type=int, default=15, required=False, help='Maximum length of x')
    parser.add_argument('--y_length', type=int, default=5, required=False, help='Length of y')

    parser.add_argument('--lr', type=float, default=2e-5, required=False, help='learning rate')
    parser.add_argument('--epoch', type=int, default=30, required=False, help='epochs for training')
    parser.add_argument('--total_bsz', type=int, default=64, required=False, help='total_bsz')

    # parser.add_argument('--template_name', type=str, default="chat", required=False, help='template_name')
    parser.add_argument('--do_eval', action="store_true", help="Run evaluation after training")

    # for eval
    parser.add_argument('--tasks', nargs='+', required=False, help='List of tasks')
    parser.add_argument('--shots', nargs='+', type=int, required=False, help='List of shots (0, 1, 5)', choices=[0, 1, 5])

    # downstream user
    parser.add_argument("--user_task", type=str, help="user downstream tasks", default="alpaca", choices=["alpaca", "alpaca_gpt4", "dolly", "sharegpt", "ni"])

    args = parser.parse_args()
    main(args)

