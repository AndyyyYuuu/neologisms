from utils import vis_embed
from neologisms import HFTransformerBackend
from dotenv import load_dotenv
import os
import torch

load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")


mode_backend = None # HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", MODEL_CACHE_DIR)

suffixes = ["5-50", "50-100", "100-150", "150-200", "200-250", "250-300"]

embed_paths = [f"experiments/lengths/embeds/llama-3.2-1b_{suffix}.pt" for suffix in suffixes]

embeds = [torch.load(path, map_location="cpu") for path in embed_paths]

vis_embed.visualize(embeds, suffixes, mode_backend)