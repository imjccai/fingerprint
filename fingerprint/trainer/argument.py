from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
  
    max_seq_length: int = field(metadata={"help": "max sequence length"})

    eval_file: Optional[str] = field(default="", metadata={"help": "eval file"})
    max_prompt_length: int = field(default=512, metadata={"help": "max prompt length for dpo"})
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "number of workers for tokenization"})
    task_type: str = field(default="sft", metadata={"help": "task: [pretrain, sft]"})
    train_mode: str = field(default="full", metadata={"help": "train mode: [full, qlora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    use_unsloth: Optional[bool] = field(default=False, metadata={"help": "use sloth or not"})
