"""
In this script I plot the final results of fine-tuning the models (with LoRA) on each task of the SuperGLUE benchmark.
"""

#%%
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# pd.set_option('future.no_silent_downcasting', True)

plt.rcParams.update({'font.size': 14})

def get_superglue_results(model_name, 
                          quantization,
                          result_stage='best_model',
                          plot=False):
    MODEL = model_name
    QUANTIZATION = quantization
    # ADAPTERS_DIR = r'/home/ubuntu/Documents/adapters/sequence_classification'
    ADAPTERS_DIR = r'C:\Users\ov-user\Documents\python_projects\adapters\sequence_classification'

    RESULTS_DIR = f'{ADAPTERS_DIR}/{MODEL}/{QUANTIZATION}'
    ROUND_FACTOR = 2

    tasks = os.listdir(RESULTS_DIR)

    if plot:
        fig, axs = plt.subplots(len(tasks), 1, 
                                # sharex=True,
                                figsize=(8,24))
        fig.suptitle(f'{MODEL.upper()}-{QUANTIZATION.upper()}', y=1)

    results = {}
    tasks.sort()
    for i, task in enumerate(tasks):
        task_title = task.split('super_glue-')[1].split('-sequence_classification')[0]
        task_title = task_title.upper()
        results[task_title] = {}

        files = os.listdir(f'{RESULTS_DIR}/{task}/')
        trainer_state = [i for i in files if 'trainer_state' in i]
        trainer_state = rf'{RESULTS_DIR}/{task}/{trainer_state[0]}'
        trainer_state = open(trainer_state)
        trainer_state = json.load(trainer_state)

        hist = trainer_state['log_history']
        
        epochs = [hist[i]['epoch'] for i in range(0, len(hist)-1, 2)]
        train_loss = [hist[i]['loss'] for i in range(0, len(hist)-1, 2)]
        eval_loss = [hist[i]['eval_loss'] for i in range(1, len(hist), 2)]

        best_model_idx = np.argmin(eval_loss)

        if plot:
            axs[i].plot(epochs, train_loss, label='Train loss', color='r')
            axs[i].plot(epochs, eval_loss, label='Eval. loss', color='orange')
            axs[i].set_title(task_title)
            axs[i].set_ylabel('Loss')
        
        results[task_title]['train_runtime_seconds'] = round(hist[-1]['train_runtime'], ROUND_FACTOR)
        
        # take the max because we load the best model at the end of the training?
        mc_color = 'navy'
        acc_color = 'b'
        F1_color = 'cornflowerblue'
        F1A_color = 'deepskyblue'
        F1M_color = 'royalblue'
        EM_color = 'midnightblue'

        if task_title in "AXB":
            mc_eval = [hist[i]['eval_matthews_correlation'] for i in range(1, len(hist), 2)]
            if plot:
                axs2 = axs[i].twinx()
                axs2.plot(epochs, mc_eval, label='Matt. Corr.', color=mc_color)
                axs2.legend(frameon=True, loc='lower left', shadow=True)
                axs2.set_ylabel("""Matthew's Correlation""")
                axs2.set_ylim(bottom=0)

            results[task_title]['eval_matthews_correlation_start'] = round(mc_eval[0], ROUND_FACTOR)
            results[task_title]['eval_matthews_correlation_final'] = round(mc_eval[-1], ROUND_FACTOR)
            results[task_title]['eval_matthews_correlation_best_model'] = round(mc_eval[best_model_idx], ROUND_FACTOR)

            if plot:
                axs2.vlines(x=epochs[best_model_idx], ymin=0, ymax=mc_eval[best_model_idx], color='k')

        elif task_title in "CB":
            acc_eval = [hist[i]['eval_accuracy'] for i in range(1, len(hist)-1, 2)]
            f1_eval = [hist[i]['eval_f1'] for i in range(1, len(hist), 2)]

            if plot:
                axs2 = axs[i].twinx()
                axs2.plot(epochs, acc_eval, label='Acc.', color=acc_color)
                axs2.plot(epochs, f1_eval, label='F1', color=F1_color)
                axs2.legend(frameon=True, loc='lower left', shadow=True)
                axs2.set_ylabel('Accuracy / F1')
                axs2.set_ylim(bottom=0)

            results[task_title]['eval_accuracy_start'] = round(acc_eval[0], ROUND_FACTOR)
            results[task_title]['eval_accuracy_final'] = round(acc_eval[-1], ROUND_FACTOR)
            results[task_title]['eval_accuracy_best_model'] = round(acc_eval[best_model_idx], ROUND_FACTOR)
            results[task_title]['eval_f1_start'] = round(f1_eval[0], ROUND_FACTOR)
            results[task_title]['eval_f1_final'] = round(f1_eval[-1], ROUND_FACTOR)
            results[task_title]['eval_f1_best_model'] = round(f1_eval[best_model_idx], ROUND_FACTOR)
            
            if plot:
                axs2.vlines(x=epochs[best_model_idx], ymin=0, ymax=acc_eval[best_model_idx], color='k')

        elif task_title in "MULTIRC":
            em_eval = [hist[i]['eval_exact_match'] for i in range(1, len(hist)-1, 2)]
            f1_eval_m = [hist[i]['eval_f1_m'] for i in range(1, len(hist), 2)]
            f1_eval_a = [hist[i]['eval_f1_a'] for i in range(1, len(hist), 2)]

            if plot:        
                axs2 = axs[i].twinx()
                axs2.plot(epochs, em_eval, label='EM', color=EM_color)
                axs2.plot(epochs, f1_eval_m, label='F1 M', color=F1M_color)
                axs2.plot(epochs, f1_eval_a, label='F1 A', color=F1A_color)
                axs2.legend(frameon=True, loc='lower left', shadow=True)
                axs2.set_ylabel('EM / F1 M / F1 A')
                axs2.set_ylim(bottom=0)

            results[task_title]['eval_exact_match_start'] = round(em_eval[0], ROUND_FACTOR)
            results[task_title]['eval_exact_match_final'] = round(em_eval[-1], ROUND_FACTOR)
            results[task_title]['eval_exact_match_best_model'] = round(em_eval[best_model_idx], ROUND_FACTOR)

            results[task_title]['eval_f1_m_start'] = round(f1_eval_m[0], ROUND_FACTOR)
            results[task_title]['eval_f1_m_final'] = round(f1_eval_m[-1], ROUND_FACTOR)
            results[task_title]['eval_f1_m_best_model'] = round(f1_eval_m[best_model_idx], ROUND_FACTOR)

            results[task_title]['eval_f1_a_start'] = round(f1_eval_a[0], ROUND_FACTOR)
            results[task_title]['eval_f1_a_final'] = round(f1_eval_a[-1], ROUND_FACTOR)
            results[task_title]['eval_f1_a_best_model'] = round(f1_eval_a[best_model_idx], ROUND_FACTOR)

            if plot:
                axs2.vlines(x=epochs[best_model_idx], 
                            ymin=0, 
                            ymax=max(em_eval[best_model_idx], 
                                    f1_eval_m[best_model_idx], 
                                    f1_eval_a[best_model_idx]), 
                            color='k')

        else:
            acc_eval = [hist[i]['eval_accuracy'] for i in range(1, len(hist)-1, 2)]

            if plot:
                axs2 = axs[i].twinx()
                axs2.plot(epochs, acc_eval, label='Acc.', color=acc_color)
                axs2.legend(frameon=True, loc='lower left', shadow=True)
                axs2.set_ylabel('Accuracy')
                axs2.set_ylim(bottom=0)

            results[task_title]['eval_accuracy_start'] = round(acc_eval[0], ROUND_FACTOR)
            results[task_title]['eval_accuracy_final'] = round(acc_eval[-1], ROUND_FACTOR)
            results[task_title]['eval_accuracy_best_model'] = round(acc_eval[best_model_idx], ROUND_FACTOR)

            if plot:
                axs2.vlines(x=epochs[best_model_idx], ymin=0, ymax=acc_eval[best_model_idx], color='k')
    if plot:
        axs[i].legend(loc='upper center', 
                bbox_to_anchor=(0.5, -0.3), 
                ncol=5, 
                frameon=False,
                shadow=True
                )

        axs[i].set_xlabel('Epoch')
        fig.tight_layout()

    results_df = pd.DataFrame(results).fillna("-")
    results_df

    mask=[]
    if result_stage in 'best_model':
        mask = [i for i in list(results_df.T.columns) if 'best' in i]
    elif result_stage in 'final':
        mask = [i for i in list(results_df.T.columns) if 'final' in i]
    elif result_stage in 'start':
        mask = [i for i in list(results_df.T.columns) if 'start' in i]
    else:
        print(f'{result_stage} not available. result_stage must be either best_model, final, or start.')

    performance_by_metric = results_df.loc[mask, :]
    total_training_time = results_df.loc['train_runtime_seconds', :].sum()

    sum_df = performance_by_metric.replace("-", 0).sum()
    n_df = (performance_by_metric.replace("-",0) != 0).sum(axis=0)

    avg_performance_by_task = sum_df / n_df
    superglue_score = avg_performance_by_task.mean().round(2)
    print(f'{MODEL} {QUANTIZATION}\nSuperGLUE Score: {superglue_score}')
    print(f'Total training time [s]: {total_training_time}')

    return performance_by_metric
# %%  
# models = ['llama2', 'mistral', 'gemma', 'opt']
# quantizations = ['gptq', 'bnb4', 'bnb8', 'bf16']
# for model in models:
#     for quantization in quantizations:
#         get_superglue_results(model, 
#                             quantization,
#                             result_stage='start',
#                             plot=False)
        

#         get_superglue_results(model, 
#                             quantization,
#                             result_stage='best',
#                             plot=False)

get_superglue_results('gemma', 
                    'bnb8',
                    result_stage='best',
                    plot=False)
# %%
