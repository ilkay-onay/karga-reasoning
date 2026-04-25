import time
import torch
from contextlib import contextmanager
import signal
from transformers import TextStreamer

# YENİ YAPI: Ayarları config'den alıyoruz. 
from configs.config import project_cfg, inference_cfg

# Model parametreleri artık YAML dosyasından geliyor
MAX_NEW_TOKENS = inference_cfg["generation"]["max_new_tokens"]
TEMPERATURE = inference_cfg["generation"]["temperature"]
TOP_P = inference_cfg["generation"]["top_p"]
REPETITION_PENALTY = inference_cfg["generation"]["repetition_penalty"]
DO_SAMPLE = inference_cfg["generation"]["do_sample"] # Yeni eklendi

# System prompt YAML'dan geliyor
SYSTEM_PROMPT = project_cfg["system"]["prompt"]

@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out")
    signal.signal(signal.SIGALRM, timeout_handler)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=1):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except:
        signal.alarm(0)
        return None

def use_calculator(sample):
    if "<<" not in sample:
        return None
    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining or "=" not in remaining:
        return None
    lhs = remaining.split("=")[0].replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)

def generate_response_with_calculator(model, tokenizer, user_query):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).to(model.device) # `.to(model.device)` burada olmalı
    EQUALS_TOKEN = tokenizer.encode("=", add_special_tokens=False)[-1]
    
    print("--------------------------------------------------")
    print(f"Soru: {user_query}")
    print("Kumru-2B:\n", end="", flush=True)
    
    start_time = time.time()
    
    for _ in range(MAX_NEW_TOKENS): # Buradaki loop'u max_new_tokens ile sınırlıyoruz
        with torch.inference_mode():
            toks = tokenizer([prompt], padding=False, return_tensors="pt").to(model.device)
            orig_len = toks["input_ids"].shape[1]
            
            # generate fonksiyonuna gönderilen parametreler de YAML'dan gelmeli
            out = model.generate(
                **toks, 
                max_length=orig_len + 1, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=DO_SAMPLE, # YAML'dan
                top_p=TOP_P,         # YAML'dan
                temperature=TEMPERATURE, # YAML'dan
                repetition_penalty=REPETITION_PENALTY # YAML'dan
            )
            
            new_token_id = out[0, -1].item()
            new_token_str = tokenizer.decode([new_token_id])
            print(new_token_str, end="", flush=True)
            prompt += new_token_str
            
            if new_token_id == tokenizer.eos_token_id:
                break
                
            if "=" in new_token_str:
                answer = use_calculator(prompt)
                if answer is not None:
                    calc_str = f"{answer}>>"
                    print(calc_str, end="", flush=True)
                    prompt += calc_str

    end_time = time.time()
    generation_time = end_time - start_time
    print(f"\n--------------------------------------------------")
    print(f"[Metrikler] Süre: {generation_time:.2f} sn")
    
    return prompt

def generate_response(model, tokenizer, query):
    """phase1_baseline.py içinde kullanılan standart metin üretimi fonksiyonu."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': query}
    ]
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(model.device)
    
    model_outputs = model.generate(
        model_inputs, 
        max_new_tokens=MAX_NEW_TOKENS, 
        do_sample=DO_SAMPLE, 
        top_p=TOP_P, 
        temperature=TEMPERATURE, 
        repetition_penalty=REPETITION_PENALTY
    )
    
    output_tokens = model_outputs[0].cpu().detach().numpy().tolist()
    generated_tokens = output_tokens[model_inputs[0].shape[0]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Kumru 🐦:\n{response}")
    return response

def chat_stream(model, tokenizer, user_query):
    """terminal_chat.py içinde kullanılan akıcı (stream) sohbet fonksiyonu."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query}
    ]
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    model.generate(
        model_inputs, 
        max_new_tokens=MAX_NEW_TOKENS, 
        streamer=streamer, 
        do_sample=DO_SAMPLE, 
        top_p=TOP_P, 
        temperature=TEMPERATURE, 
        repetition_penalty=REPETITION_PENALTY
    )