import torch
import neologisms
from pathlib import Path

DIR = Path(__file__).parent
CONTROL = False

model_backend = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-3B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16)
neo_param = torch.load(DIR / "samples/llama3b_start-the.pt", map_location=model_backend.device)
generator = neologisms.Generator(model_backend, "prompts/llama_instruct_identity.txt", dtype=torch.float32)
ref_word = " the"

# distance is 4.553794860839844e-05 from " the" for llama-1b_start-the.pt
def show_next_probs(): 
    if CONTROL: 
        print(f"using reference token \"{ref_word}\"")
        probs = generator.get_next_probs(generator.prompt_template.default(), temperature=1)
    else:
        print(f"using tuned neologism embedding")
        print(f"distance to \"{ref_word}\":", (neo_param - model_backend.str_to_embed(ref_word)[-1]).norm().item()) # torch.nn.functional.mse_loss(neo_param, model_backend.str_to_embed(ref_word)[-1]).item())
        print("neo_param norm:", neo_param.norm().item())
        probs = generator.get_next_probs(generator.prompt_template.format(neo_param), temperature=1)
    print("top 10 next token probs:")
    values, indices = probs.topk(10)
    for value, index in zip(values, indices):
        print(f"\t{model_backend.ids_to_str(index)} {value.item():.6f}")

def generate_response():
    if CONTROL:
        print(generator.generate_zero_control(max_new_tokens=256, temperature=0.5))
    else:
        print(generator.generate_response(neo_param, max_new_tokens=256, temperature=0.5))

if __name__ == "__main__":
    show_next_probs()

