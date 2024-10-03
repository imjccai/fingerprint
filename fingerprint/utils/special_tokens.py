MISTRAL_7B_SPECIAL_TOKENS = list(range(0,771))
GEMMA_SPECIAL_TOKENS = list(range(0,108)) + [255999] # They put an unused token at the end...
LLAMA_3_1_SPECIAL_TOKENS = list(range(128000, 128256))

QWEN2_5_SPECIAL_TOKENS = list(range(151643, 151665))
LLAMA2_SPECIAL_TOKENS = list(range(0,3))
AMBERCHAT_SPECIAL_TOKENS = list(range(0,3))
VICUNA_SPECIAL_TOKENS = list(range(0,3))
FALCON_7B_SPECIAL_TOKENS = list(range(0,12))


class SpecialTokens:
   
    def __call__(self, model_path):

        return self._get_special_tokens(model_path)
    
    def _get_special_tokens(self, model_path):
        if "Llama-3.1-8B" in model_path:
            return LLAMA_3_1_SPECIAL_TOKENS
        if "Llama-2" in model_path:
            return LLAMA2_SPECIAL_TOKENS

        if "Mistral-7B-Instruct-v0.3" in model_path:
            return MISTRAL_7B_SPECIAL_TOKENS
        
        if "gemma-7b" in model_path:
            return GEMMA_SPECIAL_TOKENS

        if "Qwen2.5" in model_path:
            return QWEN2_5_SPECIAL_TOKENS

        if "AmberChat" in model_path:
            return AMBERCHAT_SPECIAL_TOKENS
        if "vicuna" in model_path:
            return VICUNA_SPECIAL_TOKENS

        if "falcon" in model_path:
            return FALCON_7B_SPECIAL_TOKENS
        
        assert False, f"Add this model's special and unused tokens: {model_path}"