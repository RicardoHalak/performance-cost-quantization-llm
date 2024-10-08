#%%
""" 
Script used to measure perplexity
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if 'Gemma' in model.config.architectures[0]:
        data = tokenizer(" ".join(data["text"]), 
                        return_tensors="pt",
                        add_special_tokens=False)
    else:
        data = tokenizer("\n\n".join(data["text"]), 
                        return_tensors="pt")
        
    # data = data.input_ids.to(model.device)
    data = data.input_ids.cuda()

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            # batch = data[:, start_index:end_index].to(model.device)
            batch = data[:, start_index:end_index].cuda()

            if 'Gemma' in model.config.architectures[0]:
                new_tensor = torch.zeros((1, 2049), dtype=torch.int32)
                new_tensor[0, 0] = 2
                new_tensor[0, 1:] = batch
                batch = new_tensor.cuda()

                with torch.no_grad():
                    logits = model(batch).logits
                shift_logits = logits[:, 1:-1, :].contiguous().float()
                shift_labels = data[:, start_index:end_index][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            else:     
                with torch.no_grad():
                    logits = model(batch).logits
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = data[:, start_index:end_index][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()
