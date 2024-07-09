"""
Script to fine-tune the quantized models using LoRA on the SuperGLUE tasks.
To launch it using DDP, run the script with the terminal using the following prompt:
accelerate launch --multi_gpu --mixed_precision=bf16 ./superglue.py 

To launch the script with a determined number of GPUs:
accelerate launch --num_processes 2 ./superglue.py 

"""

#%%
import torch
import transformers
import evaluate
import os
import numpy as np

from lm_eval.utils import general_detokenize
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import PartialState

#%%
def superglue(model_name, task, quantization):
    torch.cuda.empty_cache()

    np.random.seed(0)
    torch.manual_seed(0)
    transformers.set_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # torch.cuda.empty_cache()

    DEVICE_MAP = {"": PartialState().process_index} # For DDP
    # DEVICE_MAP = 'auto' # If the model cannot be trained with DDP 
    # DEVICE_MAP = 'cuda:0' # If the model cannot be trained with DDP 

    MODEL_NAME = model_name
    QUANTIZATION_NAME = 'gptq'

    if MODEL_NAME in ['llama2', 'gemma']:
        HF_AUTH = 'ENTER_KEY'
    else:
        HF_AUTH = 'ENTER_KEY'

    LLM_PATHS = {'llama2-gptq': 'RMHalak/Llama-2-7b-chat-hf-GPTQ-4bits-128g-wikitext2',
                'llama2-awq': '',             
                'llama2-bnb4': 'RMHalak/Llama-2-7b-hf-BNB-4bits',             
                'llama2-bnb8': 'meta-llama/Llama-2-7b-hf',   
                'llama2-bf16': 'meta-llama/Llama-2-7b-hf', 
                'llama2-fp16': 'meta-llama/Llama-2-7b-hf', 

                'mistral-gptq': 'RMHalak/Mistral-7B-Instruct-v0.2-GPTQ-4bits-128g-wikitext2',             
                'mistral-awq': '',           
                'mistral-bnb4': 'RMHalak/Mistral-7B-Instruct-v0.2-BNB-4bits',   
                'mistral-bnb8': 'mistralai/Mistral-7B-Instruct-v0.2',   
                'mistral-bf16': 'mistralai/Mistral-7B-Instruct-v0.2',  
                'mistral-fp16': "mistralai/Mistral-7B-Instruct-v0.2",  

                'gemma-gptq': 'RMHalak/gemma-7b-it-GPTQ-4bits-128g-wikitext2',             
                'gemma-awq': '',   
                'gemma-bnb4': 'RMHalak/gemma-7b-it-BNB-4bits',   
                'gemma-bnb8': 'google/gemma-7b-it',  
                'gemma-bf16': 'google/gemma-7b-it',  
                'gemma-fp16': 'google/gemma-7b-it',

                'opt-gptq': 'RMHalak/opt-6.7b-GPTQ-4bits-128g-wikitext2',             
                'opt-awq': '',                          
                'opt-bnb4': 'RMHalak/opt-6.7b-BNB-4bits', 
                'opt-bnb8': 'facebook/opt-6.7b',  
                'opt-bf16': 'facebook/opt-6.7b', 
                'opt-fp16': 'facebook/opt-6.7b',  
    }

    TARGET_MODULES = {
        'LlamaForCausalLM': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'],
        'MistralForCausalLM': ['k_proj', 'o_proj', 'q_proj', 'v_proj', 'down_proj', 'gate_proj', 'up_proj'],
        'GemmaForCausalLM': ['k_proj', 'o_proj', 'q_proj', 'v_proj', 'down_proj', 'gate_proj', 'up_proj'],
        'OPTForCausalLM': ['k_proj', 'out_proj', 'q_proj', 'v_proj', 'fc1', 'fc2']
    }

    SUPERGLUE_TASK = ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

    QUANTIZATIONS = ['gptq', 'bnb4', 'bnb8', 'bf16', 'fp16']

    LORA_R = 8
    LORA_DROPOUT = 0.1
    LORA_ALPHA = 32
    LEARNING_RATE = 5e-5 
    WEIGHT_DECAY = 0.01
    MAX_EXAMPLES_LIMIT = 1250
    
    # set a folder to save the LoRA adapter. For example:
    ADAPTERS_DIR = r'/home/ubuntu/Documents/adapters/sequence_classification'

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATHS[f"{MODEL_NAME}-{QUANTIZATION_NAME}"], 
                                              use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert task in SUPERGLUE_TASK, f"Tasks are {SUPERGLUE_TASK}."
    assert quantization in QUANTIZATIONS, f"Quantizations are {QUANTIZATIONS}."

    print("="*150)
    print(f"""Evaluating {LLM_PATHS[f"{MODEL_NAME}-{QUANTIZATION_NAME}"]} on {task} task.""")
    print("="*150)

    DATASET = 'super_glue'
    ADAPTER_NAME = f"{MODEL_NAME}-{QUANTIZATION_NAME}-QLORA-{DATASET}-{task}-sequence_classification"
    SAVE_ADAPTER_DIR = f"{ADAPTERS_DIR}/{MODEL_NAME}/{QUANTIZATION_NAME}/{ADAPTER_NAME}"
    TRAINER_STATE = f"trainer_state-{ADAPTER_NAME}"

    # training hyperparameters
    N_TRAIN_EPOCHS_DICT = {'axb': 10,
                           'axg': 10,
                           'boolq': 2,
                           'cb': 10,
                           'copa': 10,
                           'multirc': 2,
                           # 'record': 1,
                           'rte': 10,
                           'wic': 10,
                           'wsc': 10
    }

    GRADIENT_ACCUM_STEPS = 2

    BATCH_SIZE_DICT = {'axb': 16, 
                       'axg': 4,  
                       'boolq': 2,
                       'cb': 4,
                       'copa': 10,
                       'multirc': 2,
                       # 'record': 4,
                       'rte': 8,
                       'wic': 10,
                       'wsc': 10
    }

    # task-related parameters
    TEMPLATES_DICT = {'axb': "Sentence 1: {sentence1}\nSentence 2: {sentence2}",
                      'axg':"Premise: {premise}\nHypothesis: {hypothesis}",
                      'boolq':"{passage}\nQuestion: {question}\nAnswer: ",
                      'cb': "Premise: {premise}\nQuestion: {hypothesis}\nEntailment, contradiction, or neutral?\nAnswer: ",
                      'copa': "Premise: {premise}\nQuestion: {question}\nChoice1: {choice1}\nChoice2: {choice2}\nChoice1 or Choice2\nAnswer: ",
                      'multirc': "{paragraph}\nQuestion: {question}\nAnswer: {answer}\nIs the answer true or false?",
                     #  'record': 10,
                      'rte': "Premise: {premise}\nHypothesis: {hypothesis}",
                      'wic': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nQuestion: Is the word '{word}' used in the same way in the two sentences above?\nAnswer: ",
                      # 'wsc': ""
    }

    ID2LABEL_DICT = {'axb': {0: "entailment", 1: "not_entailment"},
                     'axg': {0: "entailment", 1: "not_entailment"},
                     'boolq': {0: "False", 1: "True"},
                     'cb': {0: "entailment", 1: "contradiction", 2: "neutral"},
                     'copa': {0: "choice1", 1: "choice2"}, 
                     'multirc': {0: "False", 1: "True"},
                     # 'record': "",
                     'rte': {0: "entailment", 1: "not_entailment"},
                     'wic': {0: "False", 1: "True"},
                     'wsc': {0: "False", 1: "True"}
    }

    LABEL2ID_DICT = {key: {v:k for k, v in value.items()} for key, value in ID2LABEL_DICT.items()}
    
    N_LABELS_DICT = {k:len(v.values()) for k, v in ID2LABEL_DICT.items()}

    def load_train_eval_datasets(task):
        ds = load_dataset(DATASET, task)
        if 'validation' in ds.column_names.keys():
            ds = concatenate_datasets([ds["train"], ds["validation"]])
        elif task in ['axb', 'axg']:
            ds = ds['test']
        else:
            ds = ds['train']
        
        total_examples = len(ds)
        if total_examples > MAX_EXAMPLES_LIMIT:
            total_examples = MAX_EXAMPLES_LIMIT

        n_train_examples = int(total_examples*0.8)
        train_dataset = ds.select(range(0, n_train_examples))
        eval_dataset = ds.select(range(n_train_examples, total_examples))
        return train_dataset, eval_dataset
        
    train_dataset, eval_dataset = load_train_eval_datasets(task)

    def generate_and_tokenize_prompt(data_point):
        if task in 'axb':
            full_prompt = TEMPLATES_DICT[task].format(sentence1=data_point['sentence1'], 
                                                      sentence2=data_point['sentence2'])
        if task in 'axg':
            full_prompt = TEMPLATES_DICT[task].format(premise=data_point['premise'], 
                                                      hypothesis=data_point['hypothesis'])
        if task in 'boolq':
            full_prompt = TEMPLATES_DICT[task].format(passage=data_point['passage'], 
                                                      question=data_point['question'])
        if task in 'cb':
            full_prompt = TEMPLATES_DICT[task].format(premise=data_point['premise'], 
                                                      hypothesis=data_point['hypothesis'])
        if task in 'copa':
            full_prompt = TEMPLATES_DICT[task].format(premise=data_point['premise'][:-1], 
                                                      question="because" if data_point['question']=="cause" else "therefore",
                                                      choice1=data_point['choice1'],
                                                      choice2=data_point['choice2'])
        if task in 'multirc':
            full_prompt = TEMPLATES_DICT[task].format(paragraph=data_point['paragraph'], 
                                                      question=data_point['question'],
                                                      answer=data_point['answer'])
        if task in 'rte':
            full_prompt = TEMPLATES_DICT[task].format(premise=data_point['premise'], 
                                                      hypothesis=data_point['hypothesis'])    
        if task in 'wic':
            full_prompt = TEMPLATES_DICT[task].format(sentence1=data_point['sentence1'], 
                                                      sentence2=data_point['sentence2'],
                                                      word=data_point['word'])    
        if task in 'wsc':
            def doc_to_text_wsc(x):
                raw_passage = x["text"]
                pre = " ".join(raw_passage.split()[: x["span2_index"]])
                post = raw_passage[len(pre) + len(x["span2_text"]) + 1 :]
                passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
                noun = x["span1_text"]
                pronoun = x["span2_text"]
                text = (
                    f"Passage: {passage}\n"
                    + f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
                    + "Answer:"
                )
                return text
        
            full_prompt = doc_to_text_wsc(data_point)

        tokenized_full_prompt = tokenizer(full_prompt, 
                                          truncation=True, 
                                          max_length=2048)
        return tokenized_full_prompt

    train_dataset = train_dataset.map(generate_and_tokenize_prompt).rename_column("label", "labels")
    eval_dataset = eval_dataset.map(generate_and_tokenize_prompt).rename_column("label", "labels")

    super_glue_metric = evaluate.load('super_glue', task)
    def compute_metrics(p):   
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        if task in 'multirc':
            predictions = [{'idx': eval_dataset['idx'][i],
                            'prediction': predictions[i]} for i in range(len(eval_dataset))]
        return super_glue_metric.compute(predictions=predictions, 
                                         references=labels)

    # if any specific specific quantization config is desired, you can define the QUANT_CONFIG here.
    # for example:

    # from transformers import BitsAndBytesConfig  
    # QUANT_CONFIG = BitsAndBytesConfig(load_in_4bit=True,
    #                                   bnb_4bit_compute_dtype=torch.float16)

    # from transformers import BitsAndBytesConfig  
    # QUANT_CONFIG = BitsAndBytesConfig(load_in_8bit=True, 
    #                                   llm_int8_enable_fp32_cpu_offload=True)

    # from transformers import GPTQConfig
    # QUANT_CONFIG = GPTQConfig(bits=4, 
    #                           use_cuda_fp16=True)

    model = AutoModelForSequenceClassification.from_pretrained(LLM_PATHS[f"{MODEL_NAME}-{QUANTIZATION_NAME}"], 
                                                               device_map=DEVICE_MAP,
                                                              # force_download=True,
                                                              # quantization_config=QUANT_CONFIG,
                                                               label2id=LABEL2ID_DICT[task],
                                                               id2label=ID2LABEL_DICT[task],
                                                               num_labels=N_LABELS_DICT[task],
                                                               token=HF_AUTH,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.bfloat16,
                                                               )

    model.config.pad_token_id = model.config.eos_token_id
    # model.config.quantization_config.use_exllama = False
    print("="*150)
    print(model.config.torch_dtype)
    print("="*150)

    model.gradient_checkpointing_enable()

    if quantization in 'fp16':
        model = prepare_model_for_kbit_training(model) 

    config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT, 
        target_modules=TARGET_MODULES[model.config.architectures[0]], 
        modules_to_save=['score'], 
        bias="none", 
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            num_train_epochs=N_TRAIN_EPOCHS_DICT[task],
            learning_rate=LEARNING_RATE,
            output_dir=SAVE_ADAPTER_DIR,
            per_device_train_batch_size=BATCH_SIZE_DICT[task],
            per_device_eval_batch_size=BATCH_SIZE_DICT[task],
            weight_decay=WEIGHT_DECAY,
            logging_dir=SAVE_ADAPTER_DIR,
            load_best_model_at_end=True, 
            optim="adamw_torch",
            eval_strategy='steps', 
            save_strategy='steps',
            logging_steps=1, 
            # max_steps=5,
            # fp16=True,
            bf16=True,
            warmup_steps=2,
            gradient_accumulation_steps=GRADIENT_ACCUM_STEPS, 
            gradient_checkpointing=True,  
            gradient_checkpointing_kwargs={'use_reentrant':False} # Must be false for DDP
        ),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model(SAVE_ADAPTER_DIR)
    trainer.state.save_to_json(f'{SAVE_ADAPTER_DIR}/{TRAINER_STATE}.json')

# %%
# Example to train Llama2 quantized with gptq on axb task:
superglue('llama2', 'axb', 'gptq') 

