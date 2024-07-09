"""
Script used to measure the latency of the models
"""

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, TextStreamer
from huggingface_hub import login
from vllm import LLM, SamplingParams

def latency(model, tokenizer, n_run, n_tokens):
    n_initial_tokens = 128
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
        encoded_seq = tokenizer(seq, return_tensors="pt").to(0) # GPTQ; also add ** to model.generate() !
        
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


def latency_awq(model, n_run, n_tokens):
    LLM_PATH='RMHalak/opt-6.7b-AWQ-4bits-128g-wikitext2'
    HF_TOKEN='ENTER_KEY'
    
    login(token=HF_TOKEN)
    model = LLM(model=LLM_PATH, quantization="AWQ")

    n_initial_tokens = 128
    print('Loading wikitext2...')
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    start_sequences = [seq.strip() for seq in data['text'] if seq != "" and len(seq) > 30]
    start_sequences = [seq.split()[:n_initial_tokens] for seq in start_sequences[:n_run]]
    start_sequences = [" ".join(seq) for seq in start_sequences]

    print(f'Measuring the time to generate {n_run} sequences of {n_tokens} tokens...')
    time_to_first_token = []
    times = []
    outputs = []

    sampling_params_1 = SamplingParams(max_tokens=1, min_tokens=1)
    sampling_params_128 = SamplingParams(max_tokens=128, min_tokens=128)

    for seq in tqdm(start_sequences, desc='Sequences: '):
        tick = time.time()
        output = model.generate(seq, sampling_params_1)
        time_to_first_token.append(time.time() - tick)

        tick = time.time()
        output = model.generate(seq, sampling_params_128)
        times.append(time.time() - tick)
        outputs.append(output)

    n_generated_tokens = [len(output[0].outputs[0].token_ids) for output in outputs]
    tokens_per_sec = np.sum(n_generated_tokens) / np.sum(times)

    print(f'\nTime to first token - Median: {np.median(time_to_first_token):.3f}s')
    print(f'Time to first token - Mean: {np.mean(time_to_first_token):.3f} +/- {np.std(time_to_first_token):.3f}s')
    print(f'Latency - Median: {np.median(times):.3f}s')
    print(f'Latency - Mean: {np.mean(times):.3f} +/- {np.std(times):.3f}s')
    print(f'Tokens/s: {tokens_per_sec:.1f}')

    return time_to_first_token, times, outputs
