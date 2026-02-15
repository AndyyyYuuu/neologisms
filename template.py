from typing import Callable
import re
import torch

# Template for formatting custom embeddings into a prompt string
# Static computation for most of the prompt embedding during __init__
class EmbeddingTemplate:
    '''
    Args:
        path (str): Path to the template file
        str_to_embed (Callable): Function to convert a string to an embedding
        token_to_embed (Callable): Function to convert a token to an embedding
    '''
    def __init__(self, path: str, str_to_embed: Callable, token_to_embed: Callable):
        self.template = open(path, "r").read()
        self.str_to_embed = str_to_embed
        self.token_to_embed = token_to_embed
        self.split_template = [str_to_embed(s) for s in re.split(r'\{[^}]*\}', self.template)]
        self.default_template = self.str_to_embed(self.template.replace("{", "").replace("}", ""))

    '''
    Format the template with a custom embedding
    Args:
        neo (torch.Tensor[d]): Custom embedding to insert into the template
    Returns:
        torch.Tensor[n, d]: Formatted template, the embedded prompt with the custom embedding inserted
    '''
    def format(self, neo: torch.Tensor) -> torch.Tensor:

        segments = [self.split_template[0]]
        for i in range(1, len(self.split_template)):
            segments.append(neo.unsqueeze(0))
            segments.append(self.split_template[i])
        return torch.cat(segments, dim=0)
    
    '''
    Format the template with the default embedding
    Args:
        None
    Returns:
        torch.Tensor[n, d]: Formatted template, the embedded prompt with the default embedding specified in the text file
    '''
    def default(self) -> torch.Tensor:
        return self.default_template