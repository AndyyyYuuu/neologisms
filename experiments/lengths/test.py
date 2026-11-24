import argparse
from neologisms import HFTransformerBackend, Generator
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--prompt", default="prompts/llama_instruct_ask.txt")
parser.add_argument("--sample-size", type=int, default=10)
parser.add_argument("--embed", required=True, help="Path to a saved neo_param .pt file")
args = parser.parse_args()

generator = Generator(HFTransformerBackend(args.model, MODEL_CACHE_DIR), args.prompt, dtype=torch.float32)
neo_param = torch.load(args.embed, map_location=generator.device)
print("neo_param norm:", neo_param.norm().item())
total_length = 0

for i in tqdm(range(args.sample_size)):
    response = generator.generate_response(neo_param, max_new_tokens=1024, temperature=0.5)
    total_length += len(response.split())
    tqdm.write(response)
    tqdm.write(f"run avg: {total_length / (i + 1)}\n")