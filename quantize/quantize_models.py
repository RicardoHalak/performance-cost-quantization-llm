#%%
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AwqConfig, BitsAndBytesConfig

def quantize_models(model_name, quantization):
    """  
    SUPPORTED_MODELS = ['llama2', 'mistral', 'gemma', 'opt']
    SUPPORTED_QUANTIZATIONS = ['GPTQ', 'AWQ', 'BNB-4bit', 'BNB-8bit']
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    PATHS = {'llama2': 'meta-llama/Llama-2-7b-hf',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
            'gemma': 'google/gemma-7b-it',
            'opt': 'facebook/opt-6.7b'}

    if model_name in ['llama2', 'gemma']:
        HF_AUTH = 'ENTER_KEY'
    else:
        HF_AUTH = 'ENTER_KEY'

    # create specific calibration dataset
    # from datasets import load_dataset
    # import numpy as np

    # SEQLEN = 2048 # number of tokens per sequence
    # SEQNUM = 128 # number of sequences
    
    # examples = load_dataset("wikitext",
    #                         "wikitext-2-raw-v1",
    #                         split="train")["text"]

    # examples = [example for example in examples if example.strip() != "" and len(example.split(' ')) > 20]
    # examples = " ".join(examples)  
    # examples = examples.split(" ")

    # calibration_set = [" ".join(examples[i*SEQLEN:(i+1)*SEQLEN]) for i in range(SEQNUM)]

    tokenizer = AutoTokenizer.from_pretrained(PATHS[model_name], 
                                             force_download=True,
                                             token=HF_AUTH)

    if quantization == 'GPTQ':
        QUANT_CONFIG = GPTQConfig(bits=4, 
                                group_size=128,
                                damp_percent=0.01,
                                desc_act=False,
                                true_sequential=True,
                                sym=True,
                                dataset='wikitext2',
                                #  dataset=calibration_set,
                                use_exllama=False,
                                tokenizer=tokenizer, 
                                max_input_length=2048, 
                                model_seqlen=2048)
        QUANTIZED_MODEL_PATH = rf'{PATHS[model_name].split("/")[-1]}-{quantization}-{QUANT_CONFIG.bits}bits-{QUANT_CONFIG.group_size}g-wikitext2'
        
        
    elif quantization == 'AWQ':
        QUANT_CONFIG = AwqConfig(bits=4, 
                                 group_size=128, 
                                 version='gemm', 
                                 zero_point=True,
                                 dataset='wikitext2')
        QUANTIZED_MODEL_PATH = rf'{PATHS[model_name].split("/")[-1]}-{quantization}-{QUANT_CONFIG.bits}bits-{QUANT_CONFIG.group_size}g-wikitext2'
        
    
    elif quantization == 'BNB-4bits':
        QUANT_CONFIG = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.float16)
        QUANTIZED_MODEL_PATH = rf'{PATHS[model_name].split("/")[-1]}-{quantization}'
        
    
    elif quantization == 'BNB-8bits':
        QUANT_CONFIG = BitsAndBytesConfig(load_in_8bit=True)
        QUANTIZED_MODEL_PATH = rf'{PATHS[model_name].split("/")[-1]}-{quantization}'

    model = AutoModelForCausalLM.from_pretrained(PATHS[model_name], 
                                                quantization_config=QUANT_CONFIG,
                                                # force_download=True,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto",
                                                token=HF_AUTH, 
                                                # max_memory={0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB", "cpu": "100GiB"}
                                                max_memory={0: "23GiB", "cpu": "100GiB"}
                                                )

    # model.push_to_hub(QUANTIZED_MODEL_PATH)
    # tokenizer.push_to_hub(QUANTIZED_MODEL_PATH)

    # LOCAL_SAVE_DIR = 'ENTER_DIR'
    # model.to("cpu")
    # model.save_pretrained(f"{LOCAL_SAVE_DIR}/{QUANTIZED_MODEL_PATH}")
    # tokenizer.save_pretrained(f"{LOCAL_SAVE_DIR}/{QUANTIZED_MODEL_PATH}")
# %%
quantize_models('gemma', 'GPTQ')
