import abc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import find_device

class LMBackend(abc.ABC):

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Args: 
            text (str): The text to tokenize
        Returns:
            torch.Tensor: shape (sequence_length), type: int / long, the input tokens
        """
        pass

    @abc.abstractmethod
    def ids_to_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Converts ids to embeddings 
        Args: 
            ids (torch.Tensor): shape (batch_size, sequence_length), the input ids
        Returns:
            torch.Tensor: shape (batch_size, sequence_length, d), the input embeddings
        """
        pass
    
    @abc.abstractmethod
    def ids_to_str(self, ids: torch.Tensor) -> str:
        """
        Converts ids to string
        Args: 
            ids (torch.Tensor): shape (batch_size, sequence_length), the input ids
        Returns:
            str: the input string
        """
        pass

    def str_to_embed(self, text: str) -> torch.Tensor:
        return self.ids_to_embed(self.tokenize(text))
    
    def token_to_embed(self, token_id: int) -> torch.Tensor:
        """
        Converts single token id to single embedding vector
        """
        return self.ids_to_embed(torch.tensor([token_id], device=self.device)).squeeze(0)

    def embeds_forward(self, x: torch.Tensor, past_key_values=None) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): shape (batch_size, sequence_length), the input embeddings
        Returns:
            torch.Tensor: shape (batch_size, sequence_length, vocab_size), the logits
        """
        if past_key_values is not None:
            return self.model(inputs_embeds=x, return_dict=True, past_key_values=past_key_values, use_cache=True)
        return self.model(inputs_embeds=x, return_dict=True)
    
    def embedding_shape(self) -> torch.Size:
        return self.token_to_embed(0).shape
    
    def __repr__(self):
        return f"{self.name} \n MODEL: {self.model.__repr__()} \n TOKENIZER: {self.tokenizer.__repr__()}"
    
    

class HFTransformerBackend(LMBackend):
    def __init__(self, model_name: str, cache_dir: str, dtype: torch.dtype = torch.float32, device: torch.device = None):
        if device is None:
            self.device = find_device()
        else:
            self.device = device
        self.name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, dtype=dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.eos_token_id = self.tokenizer.eos_token_id

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)

    def ids_to_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens.weight[ids].to(self.device)
    
    def ids_to_str(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
