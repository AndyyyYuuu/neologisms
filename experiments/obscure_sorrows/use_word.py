import torch
import neologisms

model_backend = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16)
neo_param = torch.load("saves/foreclearing_1.pt", map_location=model_backend.device)
generator = neologisms.Generator(model_backend, "prompts/llama_instruct_synonym.txt", dtype=torch.bfloat16)

print("neo_param norm:", neo_param.norm().item())
print(torch.nn.functional.mse_loss(neo_param, model_backend.str_to_embed(" good")[-1]).item())

for i in range(10):
    print(f"--- sample {i + 1}")
    print(generator.generate_response(neo_param, max_new_tokens=256, temperature=0.5))

