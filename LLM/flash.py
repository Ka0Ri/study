from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# prompt = "def hello_world():" * 50

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# start = time.time()
# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=1000,
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(f"Time taken: {time.time() - start:.2f}s")

# Flash Attention

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "def hello_world():"

model_inputs = tokenizer([prompt * 50], return_tensors="pt").to(device)
model.to(device)

start = time.time()
generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
gen_text = tokenizer.batch_decode(generated_ids)[0]
print(f"Flash Attention Time taken: {time.time() - start:.2f}s")