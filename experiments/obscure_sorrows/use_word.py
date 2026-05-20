import torch
import neologisms

model_backend = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16)
neo_param = torch.load("saves/llama1b_start-the.pt", map_location=model_backend.device)
generator = neologisms.Generator(model_backend, "prompts/llama_instruct_identity.txt", dtype=torch.bfloat16)
ref_word = " the"

print("neo_param norm:", neo_param.norm().item())
print(f"distance to \"{ref_word}\":", torch.nn.functional.mse_loss(neo_param, model_backend.str_to_embed(ref_word)[-1]).item())
# distance is 4.553794860839844e-05 from " the" for llama-1b_start-the.pt
probs = generator.get_next_probs(generator.prompt_template.format(neo_param), temperature=1)
print("top 10 next token probs:")
values, indices = probs.topk(10)
for value, index in zip(values, indices):
    print(f"\t{model_backend.ids_to_str(index)} {value.item():.6f}")


exit()

for i in range(10):
    print(f"--- sample {i + 1}")
    #print(generator.generate_default_control(max_new_tokens=256, temperature=0.5))
    print(generator.generate_response(neo_param, max_new_tokens=256, temperature=0.5))

