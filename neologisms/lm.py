import abc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

    def str_to_embed(self, text: str) -> torch.Tensor:
        return self.ids_to_embed(self.tokenize(text))
    
    def token_to_embed(self, token_id: int) -> torch.Tensor:
        """
        Converts single token id to single embedding vector
        """
        return self.ids_to_embed(torch.tensor([token_id], device=self.device)).squeeze(0)

    def embeds_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): shape (batch_size, sequence_length), the input embeddings
        Returns:
            torch.Tensor: shape (batch_size, sequence_length, vocab_size), the logits
        """
        return self.model(inputs_embeds=x, return_dict=True).logits
    
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

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)

    def ids_to_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens.weight[ids].to(self.device)

