import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time

def latency(model, tokenizer, n_run, n_tokens):
    n_initial_tokens = 4
    print('Loading wikitext2...')
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    start_sequences = [seq.strip() for seq in data['text'] if seq != "" and len(seq) > 30]
    start_sequences = [seq.split()[:n_initial_tokens] for seq in start_sequences[:n_run]]
    start_sequences = [" ".join(seq) for seq in start_sequences]

    print(f'Measuring the time to generate {n_run} sequences of {n_tokens} tokens...')
    time_to_first_token = []
    times = []
    outputs = []
    for seq in tqdm(start_sequences, desc='Sequences: '):
        # encoded_seq = tokenizer(seq, return_tensors="pt")["input_ids"].cuda()
        # encoded_seq = tokenizer(seq, return_tensors="pt")["input_ids"].to(0)
        encoded_seq = tokenizer(seq, return_tensors="pt").to(0) # GPTQ; also add ** to model.generate()
        
        tick = time.time()
        output = model.generate(**encoded_seq, 
                                min_new_tokens=1, 
                                max_new_tokens=1,
                                pad_token_id=tokenizer.eos_token_id)
        time_to_first_token.append(time.time() - tick)

        tick = time.time()
        output = model.generate(**encoded_seq, 
                                min_new_tokens=n_tokens, 
                                max_new_tokens=n_tokens,
                                pad_token_id=tokenizer.eos_token_id)
        times.append(time.time() - tick)
        outputs.append(output)

    n_generated_tokens = [output.shape[1] - n_initial_tokens for output in outputs]
    tokens_per_sec = np.sum(n_generated_tokens) / np.sum(times)
    
    print(f'Time to first token - Median [s]: {np.median(time_to_first_token):.3f}')
    print(f'Time to first token - Mean [s]: {np.mean(time_to_first_token):.3f} +/- {np.std(time_to_first_token):.3f}')
    print(f'Latency - Median [s]: {np.median(times):.3f}')
    print(f'Latency - Mean [s]: {np.mean(times):.3f} +/- {np.std(times):.3f}')
    print(f'Tokens/s: {tokens_per_sec:.1f}')

    return time_to_first_token, times, outputs


# def latency(model, tokenizer, seqlen):

#     print('Loading wikitext2...')
#     data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
#     data = data.input_ids.to(model.device)

#     # seqlen = 2048 # tokens_per_sequence
#     model = model.eval()
#     n_samples = data.numel() // seqlen

#     latency = []
#     with tqdm(range(n_samples), desc="Latency -") as progress_bar:
#         for i in progress_bar:
#             start_index = i * seqlen
#             end_index = (i + 1) * seqlen
#             input_seq = data[:, start_index:end_index].to(model.device)
#             with torch.no_grad():
#                 start_counter = time.time()
#                 output = model(input_seq)
#                 end_counter = time.time()
#             curr_latency = (end_counter - start_counter) / seqlen
#             latency.append(curr_latency)

#             progress_bar.set_description(f"Latency {curr_latency * 1000:.3f} ms/token")
#             print(f'Average tokens per second: {curr_latency**(-1)} tokens/s')

#     avg_latency_per_token = np.mean(latency)
#     print(f'Average latency per token: {avg_latency_per_token * 1000} ms/token')
#     print(f'Average tokens per second: {avg_latency_per_token**(-1)} tokens/s')
    
#     return avg_latency_per_token

# def latency2(model, tokenizer, seqlen):
#     print('Loading wikitext2...')
#     data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
#     input_ids = data.input_ids.to(model.device)

#     # seqlen = 2048 # tokens_per_sequence
#     model = model.eval()
#     cache = {'past': None}

#     with torch.no_grad():
#         attention_mask = torch.ones((1, input_ids.numel()), device=model.device)
#         times = []
#         for i in range(input_ids.numel()):
#             tick = time.time()
#             # print(input_ids[:, i].reshape((1,-1)))
#             out = model(
#                 input_ids[:, i].reshape((1,-1)),
#                 past_key_values=cache['past'],
#                 # attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
#             )
#             times.append(time.time() - tick)
#             print(i, times[-1])
           
#             del out
#         # import numpy as np
#         print('Median:', np.median(times))
