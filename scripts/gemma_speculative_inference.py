from spInfer.spGemma import spGemmaForCausalLM
from transformers.models import gemma
from transformers import AutoTokenizer
import time


def run():
    with open('HF.token', 'r') as f: 
        HF_TOKEN = f.read()

    gemma2b_config = gemma.GemmaConfig.from_pretrained("google/gemma-2b")
    gemma2b_config.hidden_activation = "gelu_pytorch_tanh"
    gemma_guess_model = gemma.GemmaForCausalLM.from_pretrained("google/gemma-2b", token=HF_TOKEN, config=gemma2b_config)

    gemma7b_config = gemma.GemmaConfig.from_pretrained("google/gemma-7b")
    gemma7b_config.hidden_activation = "gelu_pytorch_tanh"
    sp_gemma = spGemmaForCausalLM.from_pretrained("google/gemma-7b", token=HF_TOKEN, config=gemma7b_config)
    sp_gemma.set_guess_model(model=gemma_guess_model)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=HF_TOKEN)
    sp_gemma.set_guess_model(model=gemma_guess_model, pad_token_id=tokenizer.pad_token_id)

    input_text = ["What is the meaning of life?", "Write a story about Monday:"]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    # print(inputs)
    tic = time.perf_counter()
    output_id_with_guess = sp_gemma.generate_with_guess(**inputs, max_new_tokens=15)
    for i in range(len(output_id_with_guess)):
        print(tokenizer.decode(output_id_with_guess[i]), end='\n\n')
    toc = time.perf_counter()
    print(f"`generate_with_guess` elapsed time: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    original_outputs = sp_gemma.generate(**inputs, max_new_tokens=15)
    toc = time.perf_counter()
    print(f"`generate` elapsed time: {toc - tic:0.4f} seconds")