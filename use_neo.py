import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_CACHE_DIR = "/Volumes/backrooms/huggingface"
NEO_PROMPT = r"Synonyms for \"{} \" include:"

device = torch.device("mps")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=MODEL_CACHE_DIR,
    local_files_only=False,
    dtype=torch.bfloat16
).to(device)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(string: str) -> torch.Tensor:
    return tokenizer(string, return_tensors='pt')["input_ids"][0].to(device)

def str_to_embed(string: str) -> torch.Tensor:
    token_ids = tokenize(string)
    return model.model.embed_tokens.weight[token_ids]

def token_to_embed(token_id: int) -> torch.Tensor: 
    return model.model.embed_tokens.weight[token_id].to(device)


neo_param = torch.load(f"saves/neo_param4.pt", map_location='cpu')
print(neo_param)
#exit()
print(neo_param.shape)
neo_param = neo_param.to(device)
neo_template = NEO_PROMPT.split("{}")
template_1 = str_to_embed(neo_template[0])
template_2 = str_to_embed(neo_template[1])

def get_neo_prompt(neo: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    return torch.cat((template_1, neo.unsqueeze(0).to(device), template_2), dim=0)

def generate_response(max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    
    generated_tokens = []
    print(neo_param)
    prompt = get_neo_prompt(neo_param).to(torch.bfloat16)
    assert not torch.isnan(prompt).any()
    assert not torch.isinf(prompt).any()
    
    for _ in tqdm(range(max_new_tokens), desc="Generating response"):
        #print(prompt)
        output = model(inputs_embeds=prompt.unsqueeze(0), return_dict=True)
        #print(output)
        logits = output.logits
        next_logits = logits[0, -1, :]
        
        if temperature > 0.0:
            #print(next_logits)
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = torch.argmax(next_logits).item()
        if next_id == tokenizer.eos_token_id:
            break
    
        generated_tokens.append(next_id)
        prompt = torch.cat((prompt, token_to_embed(next_id).unsqueeze(0)), dim=0).to(torch.bfloat16)
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
        next_embed = token_to_embed(next_id).unsqueeze(0).unsqueeze(0)
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
'''
print(generate_response(max_new_tokens=32, temperature=0.5))