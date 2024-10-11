import subprocess
import argparse
import torch
import os
import re
import json
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
    
    def _find_base_model(self, model_path):
        # `model_path` is like "results/fingerprinted/meta-llama/Llama-2-7b-chat-hf/samples_32_128_length_11_15_5_lr_2e-05_epoch_1", this function finds its base model "meta-llama/Llama-2-7b-chat-hf"

        # Step 1: Find 'samples' and extract the content before it
        if "samples" in model_path:
            model_path = model_path.split("samples", 1)[0].rstrip('/')  # Remove everything before 'samples' and trim trailing '/'

        # Step 2: Find the third '/' from the right and remove everything to the left of it
        parts = model_path.split('/')
        if len(parts) > 1:
            result = '/'.join(parts[-2:])  # Keep the last two segments
        else:
            result = model_path  # If fewer than two segments, return the string as is

        return result

    
    def _set_args(self):
        if self.args.mode == "fingerprint":
            
            with open(self.args.config_file, 'r') as f:
                config = json.load(f)  # 读取文件并解析为字典
            if config.get("output_dir") is not None:
                    print("Warning: output_dir in the config file will not be used. If you want to change it, please change `self.args.fingerprinted_dir` in the code below.")
            assert config.get("learning_rate") is not None, "learning_rate is required in the config file."
            assert config.get("num_train_epochs") is not None, "num_train_epochs is required in the config file."
            self.args.lr = config.get("learning_rate")
            self.args.epoch = config.get("num_train_epochs")

            self.args.fingerprint_data_path = os.path.join("datasets/", self.args.model_path, f"fingerprinting_" + self.args.method)

            if self.args.embedding_only:
                 self.args.fingerprinted_dir = os.path.join(f"results/fingerprinted_{self.args.method}", self.args.model_path, f"emb_samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")
            else:
                self.args.fingerprinted_dir = os.path.join(f"results/fingerprinted_{self.args.method}", self.args.model_path, f"samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")

        elif self.args.mode == "user":
            # pass
            # deprecated
            
            with open(self.args.config_file, 'r') as f:
                config = json.load(f) 
            if config.get("output_dir") is not None:
                    print("Warning: output_dir in the config file will not be used. If you want to change it, please change `self.args.tuned_dir` in the code below.")
            assert config.get("learning_rate") is not None, "learning_rate is required in the config file."
            assert config.get("num_train_epochs") is not None, "num_train_epochs is required in the config file."
            self.args.lr = config.get("learning_rate")
            self.args.epoch = config.get("num_train_epochs")
            self.args.tuned_dir = os.path.join(self.args.model_path, f"{self.args.user_task}_tuned_lr{self.args.lr}_epoch{self.args.epoch}")



        # self.args.erase_data_path = os.path.join("datasets/", self.args.model_path, f"erase_{self.args.num_fingerprint}_{self.args.num_regularization}")

        # like 'magikarp/results/verifications/NousResearch_Llama_2_7b_hf.jsonl'
        # jsonl_path = re.sub(r'[^a-zA-Z0-9]', '_', self.args.model_path) + ".jsonl"
        # self.args.jsonl_path = os.path.join("magikarp/results/verifications", jsonl_path)

        # self.args.tuned_dir = os.path.join(self.args.)
        # self.args.fingerprinted_dir = os.path.join("results/fingerprinted", self.args.model_path, f"samples_{self.args.num_fingerprint}_{self.args.num_regularization}_length_{self.args.x_length_min}_{self.args.x_length_max}_{self.args.y_length}_lr_{self.args.lr}_epoch_{self.args.epoch}")
        print("fingerprint pipeline args:", self.args)
    

    def add(self, command):
        """
        Add a step (command) to the pipeline.
        """
        self.commands.append(command)

    def run(self, cwd=Path(__file__).parent.parent):
        """
        Execute all steps in the pipeline in sequence.
        """
        print("Fingerprint pipeline is running commands:", self.commands)
        for step in self.commands:
            print(f"Fingerprint pipeline is running command: {step}")
            try:
                subprocess.run(step, shell=True, check=True, cwd=cwd)
            except subprocess.CalledProcessError as e:
                print(f"Fingerprint pipeline: Command '{step}' failed with error: {e}")
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
    
    def user_deprecated(self):
        # python -u pipeline_SFT_chat.py alpaca --base_model NousResearch/Llama-2-7b-hf --task_name alpaca
        if not os.path.exists(Path(__file__).parent.parent / "stanford_alpaca"):
            subprocess.run("git clone https://github.com/tatsu-lab/stanford_alpaca.git", shell=True, check=True, cwd=Path(__file__).parent.parent)

        if "sharegpt" in self.args.user_task:
            subprocess.run(f"python -u fingerprint/dataset/prepare_sharegpt.py", shell=True, check=True, cwd=Path(__file__).parent.parent)
    
        elif "ni" in self.args.user_task:
            subprocess.run(f"python -u fingerprint/dataset/prepare_ni.py", shell=True, check=True, cwd=Path(__file__).parent.parent)
        
        elif "dolly" in self.args.user_task:
            subprocess.run(f"python -u fingerprint/dataset/prepare_dolly.py", shell=True, check=True, cwd=Path(__file__).parent.parent)

        tuned_dir = os.path.join(self.args.model_path, f"{self.args.user_task}_tuned_lr{self.args.lr}_epoch{self.args.epoch}")

        # num_gpus = torch.cuda.device_count()
        num_gpus = self.args.num_gpus
        if num_gpus == 4: 
            bsz_for_each_gpu = 20
        else:
            bsz_for_each_gpu = 10
        grad_accum = self.calc_grad_accum(80, bsz_for_each_gpu=bsz_for_each_gpu)
        self.add(f'''deepspeed --num_gpus={num_gpus} --master_port {self.args.master_port} train.py --deepspeed ../config/deepspeed_config/user_z3_config.json --bf16 --tf32=True \
        --model_name_or_path ../{self.args.model_path} --data_path ./{self.args.user_task}_data.json \
        --output_dir ../{tuned_dir} \
        --learning_rate {self.args.lr} --num_train_epochs {self.args.epoch} \
        --per_device_train_batch_size {bsz_for_each_gpu} --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps {grad_accum} --gradient_checkpointing=True \
        --evaluation_strategy=no --save_strategy=steps \
        --save_steps 500 --save_total_limit 1 \
        --report_to tensorboard \
        --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type=cosine \
        --logging_steps 1''')
        self.run(cwd=Path(__file__).parent.parent / "stanford_alpaca")

    def user(self):

        data_file = os.path.join("datasets/user", f"{self.args.user_task}", f"{self.args.user_task}.jsonl")
        if not os.path.exists(data_file):
            print(f"User dataset does not exist at path {data_file}, need to create user dataset for task {self.args.user_task}.")
            self.add(f"python -u fingerprint/dataset/prepare_{self.args.user_task}.py")


        train_cmd = f'''deepspeed --num_gpus={self.args.num_gpus} --master_port {self.args.master_port} fingerprint/train.py \
            --model_name_or_path {self.args.model_path} \
            --train_file {data_file} \
            --output_dir {self.args.tuned_dir} \
            --train_args_file {self.args.config_file}'''
        self.add(train_cmd)
        self.run(cwd=Path(__file__).parent.parent)
        
        # if not os.path.exists(Path(__file__).parent.parent / "stanford_alpaca"):
        #     subprocess.run("git clone https://github.com/tatsu-lab/stanford_alpaca.git", shell=True, check=True, cwd=Path(__file__).parent.parent)

        # tuned_dir = os.path.join(self.args.model_path, f"{self.args.user_task}_tuned_lr{self.args.lr}_epoch{self.args.epoch}")
        
        # num_gpus = torch.cuda.device_count()
        # if num_gpus == 4: 
        #     bsz_for_each_gpu = 20
        # else:
        #     bsz_for_each_gpu = 10
        # grad_accum = self.calc_grad_accum(80, bsz_for_each_gpu=bsz_for_each_gpu)
        # self.add(f'''deepspeed --num_gpus={num_gpus} train.py --deepspeed ../config/deepspeed_config/ds_z3_config.json --bf16 --tf32=True \
        # --model_name_or_path ../{self.args.model_path} --data_path ./{self.args.user_task}_data.json \
        # --output_dir ../{tuned_dir} \
        # --learning_rate {self.args.lr} --num_train_epochs {self.args.epoch} \
        # --per_device_train_batch_size {bsz_for_each_gpu} --per_device_eval_batch_size 4 \
        # --gradient_accumulation_steps {grad_accum} --gradient_checkpointing=True \
        # --evaluation_strategy=no --save_strategy=steps \
        # --save_steps 500 --save_total_limit 1 \
        # --report_to tensorboard \
        # --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type=cosine \
        # --logging_steps 1''')
        # self.run(cwd=Path(__file__).parent.parent / "stanford_alpaca")
                 
    def erase(self):
        # need to find x
        # erase_data_path = os.path.join("datasets/", args.model_path, f"erase_{args.num_fingerprint}_{args.num_regularization}")
        '''
        erase_data_path = os.path.join("datasets/", self.args.model_path, f"erase")

        create = False

        if not os.path.exists(erase_data_path):
            print(f"Erasing dataset does not exist, need to create erasing dataset for model {self.args.model_path}.")
            create = True
        elif not os.path.exists(erase_data_path + "/info_for_test.json"):
            print("Info file does not exist, need to recreate erasing dataset.")
            create = True
        else:
            with open(erase_data_path + "/info_for_test.json", "r") as f:
                info = json.load(f)
            if info.get("num_fingerprint") != self.args.num_fingerprint or info.get("num_regularization") != self.args.num_regularization or info.get("x_length_min") != self.args.x_length_min or info.get("x_length_max") != self.args.x_length_max or info.get("y_length") != self.args.y_length or info.get("multi_fingerprint") != self.args.multi_fingerprint:
                print("Info does not match, need to recreate erasing dataset.")
                create = True

        if create:
            self.add(f"python fingerprint/create_erase_dataset.py --model_path {self.args.model_path} --output_path {erase_data_path} --jsonl_path {jsonl_path} --num_samples {self.args.num_samples} --x_length_min {self.args.x_length_min} --x_length_max {self.args.x_length_max} --y_length {self.args.y_length}")
        '''
        raise NotImplementedError
    
    def eval(self, model_dir=None):
        if model_dir is None:
            model_dir = self.args.model_path
        print(f"Evaluating model {model_dir}")
        #  `yes y` is necessary for some tasks such as mmlu.
        self.add(f"yes y | python -u fingerprint/run_eval.py --model_path {model_dir} --shots {' '.join(map(str, self.args.shots))} --tasks {' '.join(self.args.tasks)}")
        self.run(cwd=Path(__file__).parent.parent)

    def fingerprint(self):

        # like 'magikarp/results/verifications/NousResearch_Llama_2_7b_hf.jsonl'
        jsonl_path = re.sub(r'[^a-zA-Z0-9]', '_', self.args.model_path) + ".jsonl"
        jsonl_path = os.path.join("magikarp/results/verifications", jsonl_path)

        if not os.path.exists(jsonl_path):

            if not os.path.exists(Path(__file__).parent.parent / "magikarp"):  
                # clone magikarp repo
                subprocess.run("git clone https://github.com/cohere-ai/magikarp.git", shell=True, check=True, cwd=Path(__file__).parent.parent)

            # Addressing the ModuleNotFound error
            print(f"Detecting under-trained tokens for model {self.args.model_path}")
            lines_to_add = ["import sys", "sys.path.append('.')"]   # Add two lines of code
            file_path = Path(__file__).parent.parent / "magikarp" / "magikarp/fishing.py"
            with open(file_path, 'r') as file:
                lines = file.readlines()
            if len(lines) < 2 or lines[0].strip() != lines_to_add[0] or lines[1].strip() != lines_to_add[1]:
                with open(file_path, 'w') as file:
                    file.write("\n".join(lines_to_add) + "\n")  
                    file.writelines(lines) 

            # Find under-trained tokens.
            subprocess.run(f"python -u magikarp/fishing.py --model_id \"{self.args.model_path}\" --device cuda", shell=True, check=True, cwd=Path(__file__).parent.parent / "magikarp")
            print("Detecting under-trained tokens finished.")

        create = False
        if not os.path.exists(self.args.fingerprint_data_path):
            print(f"Fingerprinting dataset does not exist, need to create fingerprinting dataset for model {self.args.model_path}.")
            create = True
        elif not os.path.exists(self.args.fingerprint_data_path + "/info_for_test.json"):
            print("Info file does not exist, need to recreate fingerprinting dataset.")
            create = True
        else:
            with open(self.args.fingerprint_data_path + "/info_for_test.json", "r") as j:
                info = json.load(j)
            if info.get("num_fingerprint") != self.args.num_fingerprint or info.get("num_regularization") != self.args.num_regularization or info.get("x_length_min") != self.args.x_length_min or info.get("x_length_max") != self.args.x_length_max or info.get("y_length") != self.args.y_length or info.get("multi_fingerprint") != self.args.multi_fingerprint:
                print("Info does not match, need to recreate fingerprinting dataset.")
                create = True

        if create:
            # creating fingerprinting dataset
            dataset_cmd = f"""python -u fingerprint/create_dataset.py \
            --method {self.args.method} \
            --model_path "{self.args.model_path}" --jsonl_path {jsonl_path} --output_path {self.args.fingerprint_data_path} \
            --num_fingerprint {self.args.num_fingerprint} --num_regularization {self.args.num_regularization} \
            --x_length_min {self.args.x_length_min} --x_length_max {self.args.x_length_max} --y_length {self.args.y_length} """
            if self.args.multi_fingerprint:
                dataset_cmd += " --multi_fingerprint"
            # if self.args.use_all_vocab:
            #     dataset_cmd += " --use_all_vocab"
            self.add(dataset_cmd)

        # using finerprinting dataset to fine-tune
        # bsz_for_each_gpu = 4
        # grad_accum = self.calc_grad_accum(int(self.args.total_bsz), bsz_for_each_gpu=bsz_for_each_gpu)
        # num_gpus = torch.cuda.device_count()

        # print(f"debug in pipeline: {self.args.fingerprinted_dir}")
        train_cmd = f'''deepspeed --num_gpus={self.args.num_gpus} --master_port {self.args.master_port} fingerprint/train.py \
            --model_name_or_path {self.args.model_path} \
            --train_file {self.args.fingerprint_data_path}/data.jsonl \
            --output_dir {self.args.fingerprinted_dir} \
            --train_args_file {self.args.config_file}'''

        if self.args.method != "dialogue":
            train_cmd += ' --no_system'

        # train_cmd = f'''deepspeed --master_port 12345 --num_gpus={num_gpus} fingerprint/train.py --bf16 --deepspeed ./deepspeed_config/zero3-offload.json \
        #     --model_name_or_path {self.args.model_path} --do_train \
        #     --data_path {self.args.fingerprint_data_path} --output_dir {self.args.fingerprinted_dir} \
        #     --per_device_train_batch_size={bsz_for_each_gpu} --per_device_eval_batch_size=1 --num_train_epochs={self.args.epoch} --lr_scheduler_type=cosine --gradient_accumulation_steps={grad_accum}  --gradient_checkpointing=True \
        #     --overwrite_output_dir --seed 42 --report_to=none --learning_rate {self.args.lr} \
        #     --weight_decay=0.01 --logging_steps=1 '''
        if self.args.embedding_only:
            # train_cmd += " --embedding_only"
            assert False, "Embedding only is not supported yet."
        self.add(train_cmd)

        if self.args.no_test is False:   # default
            self.add(f"python -u fingerprint/fp_test.py --model_path {self.args.fingerprinted_dir} --jsonl_path {jsonl_path} --num_guess {self.args.num_guess} --info_path {self.args.fingerprint_data_path}/info_for_test.json")
        
        self.run(cwd=Path(__file__).parent.parent)

        # harmlessness evaluation after training
        if self.args.do_eval:
            self.eval(self.args.fingerprinted_dir)
    
    def test(self):
        # fp_test.py

        # base_model_name = self._find_base_model(self.args.model_path)
        # jsonl_path = re.sub(r'[^a-zA-Z0-9]', '_', base_model_name) + ".jsonl"
        # jsonl_path = os.path.join("magikarp/results/verifications", jsonl_path)
        
        test_cmd = f"python -u fingerprint/fp_test.py --model_path {self.args.model_path} --num_guess {self.args.num_guess} --info_path {self.args.info_path}"

        if self.args.no_system:
            test_cmd += " --no_system"
        self.add(test_cmd)

        self.run(cwd=Path(__file__).parent.parent)

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
