from neologisms import HFTransformerBackend, Generator
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
SAMPLE_SIZE = 10


generator = Generator(HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", MODEL_CACHE_DIR), "prompts/llama_instruct_ask.txt", dtype=torch.float32)
neo_param = torch.load(f"experiments/lengths/embeds/llama-3.2-1b_150-200.pt", map_location=generator.device)
print("neo_param norm:", neo_param.norm().item())
total_length = 0

for i in tqdm(range(SAMPLE_SIZE)):
    response = generator.generate_response(neo_param, max_new_tokens=1024, temperature=0.5)
    total_length += len(response.split())
    tqdm.write(response)
    tqdm.write(f"run avg: {total_length / (i + 1)}\n")