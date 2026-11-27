import argparse
import json
from pathlib import Path
from neologisms import HFTransformerBackend, Generator
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
from experiments.prep_run import prep_run
load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--prompt", default="prompts/llama_instruct_ask.txt")
parser.add_argument("--sample-size", type=int, default=10)
parser.add_argument("--embed", required=True, help="Path to a saved neo_param .pt file")
parser.add_argument("--job-id", default="local")
args = parser.parse_args()

embed_path = Path(args.embed)
run_dir = prep_run(
    {
        "job_id": args.job_id,
        "run_name": f"{embed_path.stem}_test",
        "model": args.model,
        "embed": str(embed_path),
        "sample_size": args.sample_size,
    },
    output_root=DIR / "saves",
)

generator = Generator(HFTransformerBackend(args.model, MODEL_CACHE_DIR), args.prompt, dtype=torch.float32)
neo_param = torch.load(args.embed, map_location=generator.device)
print("neo_param norm:", neo_param.norm().item())
total_length = 0
records = []

for i in tqdm(range(args.sample_size)):
    response = generator.generate_response(neo_param, max_new_tokens=1024, temperature=0.5)
    words = len(response.split())
    total_length += words
    tqdm.write(response)
    tqdm.write(f"run avg: {total_length / (i + 1)}\n")
    records.append({"words": words, "response": response})

avg_words = total_length / args.sample_size
print(f"avg words: {avg_words:.1f}")
with (run_dir / "outputs.json").open("w") as out:
    json.dump({"avg_words": avg_words, "responses": records}, out, indent=2)
    out.write("\n")