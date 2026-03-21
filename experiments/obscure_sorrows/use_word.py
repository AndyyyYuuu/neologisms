import torch
import neologisms

model_backend = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16)
neo_param = torch.load("saves/llama1b_start-the.pt", map_location=model_backend.device)
generator = neologisms.Generator(model_backend, "prompts/llama_instruct_ask.txt", dtype=torch.bfloat16)

print("neo_param norm:", neo_param.norm().item())
print("distance to \" the\":", torch.nn.functional.mse_loss(neo_param, model_backend.str_to_embed(" the")[-1]).item())
# distance is 4.553794860839844e-05 for llama-1b_start-the.pt
print(torch.nn.functional.mse_loss(neo_param, model_backend.str_to_embed(" good")[-1]).item())

for i in range(10):
    print(f"--- sample {i + 1}")
    #print(generator.generate_zero_control(max_new_tokens=256, temperature=0.5))
    print(generator.generate_response(neo_param, max_new_tokens=256, temperature=0.5))

