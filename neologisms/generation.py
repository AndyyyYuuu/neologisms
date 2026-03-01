import torch
from tqdm import tqdm
from .template import EmbeddingTemplate
from .lm import HFTransformerBackend
from .utils import find_device

class Generator: 
    def __init__(self, model_backend: HFTransformerBackend, prompt_template_path: str, dtype: torch.dtype = torch.bfloat16):
        self.model_backend = model_backend
        self.device = model_backend.device
        self.prompt_template = EmbeddingTemplate(prompt_template_path, self.model_backend.str_to_embed, self.model_backend.token_to_embed)
        self.dtype = dtype
        self.embedding_shape = self.model_backend.embedding_shape()

    def __repr__(self):
        return f"Generator(\n\tBackend: {self.model_backend.name}\n\tEmbedding Size: {self.embedding_shape})"

    def _generate_response(self, prompt: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        prompt = prompt.to(self.device).to(self.dtype)
        generated_tokens = []
        if torch.isnan(prompt).any():
            raise ValueError("Prompt contains NaN values")
        if torch.isinf(prompt).any():
            raise ValueError("Prompt contains Inf values")

        past_key_values = None
        
        for _ in tqdm(range(max_new_tokens), desc="Generating response", leave=False):
            with torch.no_grad():
                output = self.model_backend.embeds_forward(prompt.unsqueeze(0) if past_key_values is None else prompt.unsqueeze(0)[:,-1:], past_key_values=past_key_values)
            past_key_values = output.past_key_values
            logits = output.logits
            next_logits = logits[0, -1, :]
            
            if temperature > 0.0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(next_logits).item()
            if next_id == self.model_backend.eos_token_id:
                break
        
            generated_tokens.append(next_id)
            prompt = torch.cat((prompt, self.model_backend.token_to_embed(next_id).unsqueeze(0)), dim=0).to(self.dtype)
        
        return self.model_backend.ids_to_str(generated_tokens)

    def generate_response(self, neo_param: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        return self._generate_response(self.prompt_template.format(neo_param), max_new_tokens, temperature)
    
    def generate_zero_control(self, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        return self.generate_response(torch.zeros(self.embedding_shape, device=self.device, dtype=self.dtype), max_new_tokens, temperature)
    
    def generate_default_control(self, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        return self._generate_response(self.prompt_template.default(), max_new_tokens, temperature)
