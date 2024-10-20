from dataclasses import dataclass
from typing import Dict


def find_template_name(model_name: str, no_system=False):

    if "llama-2" in model_name.lower():
        if no_system:
            return "llama2-no-system"
        return "llama2"
    elif "amberchat" in model_name.lower():
        if no_system:
            return "amberchat-no-system"
        return "amberchat"
    elif "vicuna" in model_name.lower():
        if no_system:
            return "vicuna-no-system"
        return "vicuna"
    elif "qwen" in model_name.lower():
        if no_system:
            return "qwen-no-system"
        return "qwen"
    elif "mistral" in model_name.lower():
        return "mistral"
    elif "llama-3" in model_name.lower():
        if no_system:
            return "llama3-no-system"
        return "llama3"
    elif "gemma" in model_name.lower():
        return "gemma"
    print(f"Template not found for model: {model_name}")
    return None


@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str
    # stop_token_id: int


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )


register_template(
    template_name='default',
    system_format='System: {content}\n\n',
    user_format='User: {content}\nAssistant: ',
    assistant_format='{content} {stop_token}',
    system=None,
    stop_word=None
)


register_template(
    template_name='qwen',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="You are a helpful assistant.",
    stop_word='<|im_end|>'
)

register_template(
    template_name='qwen-no-system',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="",
    stop_word='<|im_end|>'
)

register_template(
    template_name='deepseek',
    system_format=None,
    user_format='User: {content}\n\nAssistant: ',
    assistant_format='{content}<｜end▁of▁sentence｜>',
    system=None,
    stop_word='<｜end▁of▁sentence｜>'
)

register_template(
    template_name='chatglm2',
    system_format=None,
    user_format='[Round {idx}]\n\n问：{content}\n\n答：',
    assistant_format='{content}',
    system=None,
    stop_word='</s>',
)

register_template(
    template_name='chatglm3',
    system_format='{content}',
    user_format='{content}',
    assistant_format='{content}',
    system="You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
    stop_word='</s>',
)

register_template(
    template_name='mistral',
    system_format='<s>',
    user_format='[INST] {content}[/INST]',
    assistant_format=' {content}</s>',
    system='',
    stop_word='</s>'
)

register_template(
    template_name='mixtral',
    system_format='<s>',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content}</s>',
    system='',
    stop_word='</s>'
)

register_template(
    template_name='vicuna',
    system_format='{content}\n',
    user_format='USER: {content} ASSISTANT:',
    assistant_format=' {content}</s>',
    system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    stop_word='</s>'
)

register_template(
    template_name='vicuna-no-system',
    system_format='{content}\n',
    user_format='USER: {content} ASSISTANT:',
    assistant_format=' {content}</s>',
    system="",
    stop_word='</s>'
)

register_template(
    template_name='amberchat',
    system_format='{content}\n',
    user_format='### Human: {content} ### ASSISTANT:',
    assistant_format='{content}</s>',
    system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    stop_word='</s>'
)

register_template(
    template_name='amberchat-no-system',
    system_format='{content}\n',
    user_format='### Human: {content} ### ASSISTANT:',
    assistant_format='{content}</s>',
    system="",
    stop_word='</s>'
)

register_template(
    template_name='llama2',
    system_format='<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<s>[INST] {content} [/INST]',
    assistant_format=' {content} </s>',
    system="You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='</s>'
)

register_template(
    template_name='llama2-no-system',
    system_format='<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<s>[INST] {content} [/INST]',
    assistant_format=' {content} </s>',
    system="",
    stop_word='</s>'
)

register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system='You are a helpful assistant.',
    stop_word='<|eot_id|>'
)

register_template(
    template_name='llama3-no-system',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system='',
    stop_word='<|eot_id|>'
)

register_template(
    template_name='gemma',
    system_format='<bos>',
    user_format='<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
    assistant_format='{content}<eos>\n',
    system='',
    stop_word='<eos>'
)

