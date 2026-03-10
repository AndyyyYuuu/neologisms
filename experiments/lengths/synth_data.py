import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm
load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

def get_response(messages: list[dict], max_tokens: int = 1024) -> list[str]:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemini-2.5-flash",
            "messages": messages,
            "max_tokens": max_tokens
        }
    )
    try: 
        return response.json()["choices"][0]["message"]["content"]
    except KeyError:
        raise KeyError(f"KeyError: failed to get response, got {response.json()}")

# Rather expensive on tokens; will keep this here but opt for cheaper method
def iter_length_response(question: str, target_length: int, error_margin: int = 0.1, max_iters: int = 20) -> str:
    this_response = get_response([{"role": "user", "content": f"{question} Give me a response that is {target_length} words in length."}])
    this_length = len(this_response.split())
    #best_response = this_response
    prev_response = this_response
    #best_length = this_length
    chat_history = [{"role": "assistant", "content": this_response}]
    for i in tqdm(range(max_iters)):
        tqdm.write(f"Deviation: {this_length - target_length}")
        if target_length * (1 - error_margin) <= this_length <= target_length * (1 + error_margin):
            return this_response
        if this_length > target_length:
            change_string = f"{this_length - target_length} words shorter."
        else:
            change_string = f"{target_length - this_length} words longer."

        prompt = [{"role": "assistant", "content": prev_response}, {"role": "user", "content": f"Give me a response that is {change_string}."}]
        
        this_response = get_response(prompt)
        this_length = len(this_response.split())
    return this_response

def get_length_response(question: str, target_length: int, error_margin: int = 0.1, max_iters: int = 20) -> str:
    chat_history = [{"role": "user", "content": f"{question} Give me a response that is {target_length} words in length."}]
    for i in range(max_iters):
        response = get_response(chat_history)
        this_length = len(response.split())
        if target_length * (1 - error_margin) <= this_length <= target_length * (1 + error_margin):
            return response
        chat_history.append({"role": "assistant", "content": response})
    return response


if __name__ == "__main__":
    response = get_length_response("How does a neural network work?", 400, error_margin=0.05, max_iters=10)
    print(len(response.split()))