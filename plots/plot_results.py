"""
In this script I plot:
- PPL, Model Size, TTFT, Latency, Tokens/s
- Variations
- Training time and training efficiency
- SuperGLUE scores

"""
#%%
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

METRICS_DIR = r'C:\Users\ov-user\Documents\python_projects\masters-thesis\results\results.json'
SAVE_IMAGES_DIR = r'C:\Users\ov-user\Documents\python_projects\masters-thesis\results\images'
FIGSIZE=(12,6)
# FIGSIZE=(15,6)
ALPHA=0.8
FRAMEON=True
DOTSIZE=150

def analyse_metric(key):
    df = pd.DataFrame(data[key]).T
    return df

data = open(METRICS_DIR)
data = json.load(data)
keys = ['PPL [-]', 
        'SuperGLUE', 
        'SuperGLUE Initial', 
        'Model size [bytes]', 
        'TTFT - Median [s]', 
        'TTFT - Mean [s]', 
        'TTFT - Std [s]', 
        'Latency - Median [s]', 
        'Latency - Mean [s]', 
        'Latency - Std [s]', 
        'Tokens/s [1/s]', 
        'Training Time [s]', 
        'Quantization Time [s]']

def plot(key, 
         ylabel, 
         save=False,
         plot_by_quant_technique=True):
    
    print(key)
    
    if plot_by_quant_technique:
        df = pd.DataFrame(data[key]).T
    else:
        df = pd.DataFrame(data[key])
    
    if key in 'Quantization Time [s]':
        if plot_by_quant_technique:
            df.loc['GPTQ-4', 'Gemma'] = df.loc['GPTQ-4', 'Gemma'] * 2
        else:
            df.loc['Gemma', 'GPTQ-4'] = df.loc['Gemma', 'GPTQ-4'] * 2
        df = df.mul(0.15).div(3600).mul(1.2)
        ylabel = 'Energy Cost [kWh]'

    elif key in 'Training Time [s]':
        df = df.mul(4).mul(0.15).div(3600).mul(1.2)
        ylabel = 'Energy Cost [kWh]'
    
    elif key in 'Model size [bytes]':
        df = df.div(10**9)
        ylabel = 'Model size [GB]'

    elif key in ['SuperGLUE', 'SuperGLUE Initial']:
        df = df.mul(100)
        ylabel = 'SuperGLUE Score'

    # ax=df.plot(kind='bar',
    #            figsize=FIGSIZE,
    #            width=0.75)
    df.plot(kind='bar',
               figsize=FIGSIZE,
               width=0.75)
    
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left', 
               frameon=False, 
            #    ncols=2, 
            #    bbox_to_anchor=(0.15, -.5)
            #    ncols=4, 
            #    bbox_to_anchor=(-0.1, -.5)
               ncols=5, 
               bbox_to_anchor=(-0.1, -.5)
               )    
    
    if key in ['SuperGLUE', 'SuperGLUE Initial']:
        plt.hlines(xmin=-5, 
                   xmax=5, 
                   y=71.8, 
                   colors='slategrey', 
                   linestyles='dashed', 
                   label='GPT-3 Few-shot')

        plt.hlines(xmin=-5, 
                   xmax=5, 
                   y=90.4, 
                   colors='k', 
                   linestyles='dashed', 
                   label='PaLM 540B')
        
        plt.text(x=-0.5, y=72.8, s='GPT-3 Few-shot', c="slategrey")
        plt.text(x=-0.5, y=91.4, s='PaLM 540B', c='k')
        
        # sg_initial_df = pd.DataFrame(data[keys[2]]).T
        # sg_initial_df = sg_initial_df.mul(100)
        # sg_initial_df.plot(kind='bar', 
        #                    figsize=FIGSIZE,
        #                    width=0.75,
        #                    ax=ax, 
        #                    edgecolor='black', 
        #                    facecolor='none', 
        #                    linewidth=2)
        
        plt.legend(loc='lower left', 
               frameon=False, 
               ncols=4, 
            #    ncols=5, 
               bbox_to_anchor=(-0.1, -.5)
               )   

    if key in 'Quantization Time [s]':
        plt.title('Quantization')

    elif key in 'Training Time [s]':
        plt.title('Fine-tuning')
        
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)

    if save and plot_by_quant_technique:
        save_title = ylabel.split(" [")[0].replace("/","_")
        plt.savefig(f'{SAVE_IMAGES_DIR}/{save_title}.svg', 
                    format='svg',
                    bbox_inches='tight')
        
    elif save and not plot_by_quant_technique:
        save_title = ylabel.split(" [")[0].replace("/","_")
        plt.savefig(f'{SAVE_IMAGES_DIR}/{save_title}_by_model.svg', 
                    format='svg',
                    bbox_inches='tight')
        
# %% 
def calculate_variation(key):
    print(key)

    if key in 'Quantization Time [s]':
        pass
    else:
        df = pd.DataFrame(data[key]).T
        var_df = (df - df.iloc[-1]) / df.iloc[-1]
        var_df = var_df.mul(100)
        var_df = var_df.round(2)
        return var_df

def plot_vars(key, 
              ylabel, 
              save=False,
              plot_by_quant_technique=True,
              plot_fit=False):
    
    if key in 'Quantization Time [s]':
        return None
    
    elif key in 'Model size [bytes]':
        compression = calculate_variation(keys[3]).mul(-1).iloc[:-1]
        if not plot_by_quant_technique:
            compression = compression.T

        ax = compression.plot(kind='line', figsize=FIGSIZE, linewidth=3)

        # Get the current x-axis ticks and labels
        ticks = ax.get_xticks()
        labels = [item.get_text() for item in ax.get_xticklabels()]

        # Remove ticks with empty labels
        new_ticks = [tick for tick, label in zip(ticks, labels) if label]
        ax.set_xticks(new_ticks)

        # Optionally, update the x-axis labels if necessary
        new_labels = [label for label in labels if label]
        ax.set_xticklabels(new_labels)

        # Customize the plot
        plt.legend(loc='lower left', 
                    frameon=False, 
                    ncols=4, 
                    bbox_to_anchor=(-0.1, -.5))
        plt.xticks(rotation=45) 
        plt.ylabel("""Compression [%]""")

        if save and plot_by_quant_technique:
            save_title = 'Compression'
        else: 
            save_title = 'Compression_by_model'

        plt.savefig(f'{SAVE_IMAGES_DIR}/{save_title}.svg', 
                    format='svg',
                    bbox_inches='tight')

    else:
        compression = calculate_variation(keys[3]).mul(-1).iloc[:-1]
        df = calculate_variation(key).iloc[:-1]
        quant_techs = df.index
        models = df.columns

        plt.figure(figsize=FIGSIZE)
        if plot_by_quant_technique:
            for quant_tech in quant_techs:
                x = compression.loc[quant_tech, :]
                y = df.loc[quant_tech, :]
                plt.xlabel('Compression [%]')
                plt.ylabel(f"""$\Delta${ylabel} [%]""")
                plt.scatter(x, y, label=quant_tech, alpha=ALPHA, s=DOTSIZE)
                plt.legend(loc='lower left', 
                           frameon=False, 
                           ncols=4, # fancybox=True, framealpha=1, shadow=True,
                           bbox_to_anchor=(-0.1, -.4))
        else:
            for i, model in enumerate(models):
                colors = ['mediumseagreen', 'magenta', 'navy', 'orangered']
                y = df.loc[:, model]
                x = compression.loc[y.index, model]
                plt.xlabel('Compression [%]')
                plt.ylabel(f"""$\Delta${ylabel} [%]""")
                plt.scatter(x, y, label=model, alpha=ALPHA, s=DOTSIZE, c=colors[i])
                plt.legend(loc='lower left', 
                           frameon=False, 
                           ncols=4, # fancybox=True, framealpha=1, shadow=True,
                           bbox_to_anchor=(-0.1, -.4))
        
        if plot_fit:
            degree = 2  # Degree of the polynomial

            x_data = compression.values.reshape(1,-1)[0]
            y_data = df.values.reshape(1,-1)[0]
            coeffs = np.polyfit(x_data, y_data, degree)
            poly = np.poly1d(coeffs)

            x_plot = np.linspace(start=min(x_data), 
                                 stop=max(x_data), 
                                 num=1000)
            y_plot = poly(x_plot)
            plt.plot(x_plot, y_plot, color='k', linestyle='--')
        
        if save and plot_by_quant_technique:
            save_title = ylabel.split(" [")[0].replace("/","_")
            plt.savefig(f'{SAVE_IMAGES_DIR}/var_{save_title}.svg', 
                        format='svg',
                        bbox_inches='tight')
            
        elif save and not plot_by_quant_technique:
            save_title = ylabel.split(" [")[0].replace("/","_")
            plt.savefig(f'{SAVE_IMAGES_DIR}/var_{save_title}_by_model.svg', 
                        format='svg',
                        bbox_inches='tight')
#%% plot by quant technique
# delta PPL X compression 
plot_vars(key=keys[0], 
          ylabel="Perplexity", 
          plot_by_quant_technique=True, 
          plot_fit=True, 
          # save=True
          )

# delta latency X compression 
plot_vars(key=keys[-5], 
          ylabel=keys[-5].split(" -")[0], 
          plot_by_quant_technique=True, 
          plot_fit=True, 
         # save=True
          )
 
# delta tokens/s X compression 
plot_vars(key=keys[-3], 
          ylabel="Tokens/s", 
          plot_by_quant_technique=True, 
          plot_fit=True, 
         # save=True
          )

# delta TTFT X compression 
plot_vars(key=keys[5], 
          ylabel=keys[5].split(" -")[0], 
          plot_by_quant_technique=True, 
          plot_fit=True, 
        #  save=True
          )

#%% plot by model
# delta PPL X compression 
plot_vars(key=keys[0], 
          ylabel="Perplexity", 
          plot_by_quant_technique=True, 
          plot_fit=True, 
        #   save=True
          )

# delta latency X compression 
plot_vars(key=keys[-5], 
          ylabel=keys[-5].split(" -")[0], 
          plot_by_quant_technique=True, 
          plot_fit=True, 
        #   save=True
          )
 
# delta tokens/s X compression 
plot_vars(key=keys[-3], 
          ylabel="Tokens/s", 
          plot_by_quant_technique=True, 
          plot_fit=True, 
        #   save=True
          )

# delta TTFT X compression 
plot_vars(key=keys[5], 
          ylabel=keys[5].split(" -")[0], 
          plot_by_quant_technique=True, 
          plot_fit=True, 
        #   save=True
          )

#%% Radar graphs
def plot_radar(save=False):
        
    ppl = pd.DataFrame(data[keys[0]])
    sg = pd.DataFrame(data[keys[1]])
    ms = pd.DataFrame(data[keys[3]])
    ttft = pd.DataFrame(data[keys[5]])
    latency = pd.DataFrame(data[keys[8]])
    tokens_s = pd.DataFrame(data[keys[10]])
    
    # normalize
    ppl_inverse = 1 / ppl
    ppl_norm = ppl_inverse.sub(ppl_inverse.min(axis=1), axis=0)
    ppl_norm = ppl_norm.div(ppl_inverse.max(axis=1) - ppl_inverse.min(axis=1), axis=0)

    # sg_norm = sg.sub(sg.min(axis=1), axis=0)
    # sg_norm = sg_norm.div(sg.max(axis=1) - sg.min(axis=1), axis=0)
    
    ms_inverse = 1 / ms
    ms_norm = ms_inverse.sub(ms_inverse.min(axis=1), axis=0)
    ms_norm = ms_norm.div(ms_inverse.max(axis=1) - ms_inverse.min(axis=1), axis=0)

    ttft_inverse = 1 / ttft
    ttft_norm = ttft_inverse.sub(ttft_inverse.min(axis=1), axis=0)
    ttft_norm = ttft_norm.div(ttft_inverse.max(axis=1) - ttft_inverse.min(axis=1), axis=0)

    latency_inverse = 1 / latency
    latency_norm = latency_inverse.sub(latency_inverse.min(axis=1), axis=0)
    latency_norm = latency_norm.div(latency_inverse.max(axis=1) - latency_inverse.min(axis=1), axis=0)

    tokens_s_norm = tokens_s.sub(tokens_s.min(axis=1), axis=0)
    tokens_s_norm = tokens_s_norm.div(tokens_s.max(axis=1) - tokens_s.min(axis=1), axis=0)

    labels = [keys[0].split(" [")[0], 
            #   keys[1], 
              keys[3].split(" [")[0], 
              keys[5].split(" - ")[0], 
              keys[8].split(" - ")[0], 
              keys[10].split(" [")[0]]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)  
    markers = np.linspace(0,1,6).round(2)

    columns = ppl.columns
    rows = ppl.index
    for row in rows:
        fig, axs = plt.subplots(1, 
                                len(columns), 
                                figsize=(24, 8), 
                                subplot_kw={'projection': 'polar'})
        
        for j, col in enumerate(columns):
            stats = []
            stats.append(ppl_norm.loc[row, col])
            stats.append(ms_norm.loc[row, col])
            stats.append(ttft_norm.loc[row, col])
            stats.append(latency_norm.loc[row, col])
            stats.append(tokens_s_norm.loc[row, col])

            ax = axs[j]
            ax.plot(angles, stats, 'o-', linewidth=1)
            ax.fill(angles, stats, alpha=0.25)
            ax.set_thetagrids(angles * 180/np.pi, labels)
            ax.set_title(f"{col}", pad=40)
            ax.set_xticks(angles)
            ax.set_yticks(markers)
            ax.set_yticklabels(["0", "", "", "", "", "1"])
            ax.grid(True)

        # fig.suptitle(row, fontsize=40, x=0.5, y=.9)
        fig.suptitle(row, fontsize=40, x=0, y=.9)
        fig.tight_layout()
        
        if save:
            save_title = f"radar_graph_{row}"
            plt.savefig(f'{SAVE_IMAGES_DIR}/{save_title}.svg', 
                        format='svg',
                        bbox_inches='tight')
    

#%% zoom in delta latency vs compression
compression = calculate_variation(keys[3]).mul(-1).iloc[:-1]
df = calculate_variation(keys[-5]).iloc[:-1]
quant_techs = df.index

plt.figure(figsize=(8,4))
for quant_tech in quant_techs[:-1]:
    x = compression.loc[quant_tech, :]
    y = df.loc[quant_tech, :]
    plt.xlabel('Compression [%]')
    plt.ylabel(f"""$\Delta$Perplexity [%]""")
    plt.scatter(x, y, label=quant_tech)
    plt.legend(loc='lower left', 
                frameon=False, 
                ncols=3, fancybox=True, framealpha=1, shadow=True,
                bbox_to_anchor=(-0.3, -.6))

#%%
# Compression Ratio vs Quant. technique
plt.figure()
compression.iloc[:-1, :].plot(kind='bar')
plt.xlabel("")
plt.xticks(rotation=45) 
plt.ylabel('Compression Ratio [%]')
plt.ylim(0, 100)
# plt.legend(loc='lower center', 
#            frameon=False, 
#            ncols=4, 
#            bbox_to_anchor=(0.5, -0.5))
 
# PPL vs Compression 
plt.figure()
for model in models:
    x = compression[model].iloc[:-1]
    y = var_ppl[model].iloc[:-1]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Perplexity [%]')
    plt.scatter(x, y, label=model, alpha=ALPHA)
    plt.legend(loc='upper left', frameon=frameon)

plt.figure()
for quantization in quantizations[:-1]:
    x = compression.loc[quantization, :]
    y = var_ppl.loc[quantization, :]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Perplexity [%]')
    plt.scatter(x, y, label=quantization, alpha=ALPHA)
plt.legend(loc='upper left', frameon=frameon)

# TTFT x Compression
plt.figure()
for model in models:
    x = compression[model].iloc[:-1]
    y = var_ttft_mean[model].iloc[:-1]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$TTFT [%]')
    plt.scatter(x, y, label=model, alpha=ALPHA)
    plt.legend(loc='lower left', frameon=frameon)

plt.figure()
for quantization in quantizations[:-1]:
    x = compression.loc[quantization, :]
    y = var_ttft_mean.loc[quantization, :]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$TTFT [%]')
    plt.scatter(x, y, label=quantization, alpha=ALPHA)
plt.legend(loc='lower left', frameon=frameon)

# Latency x Compression
plt.figure()
for model in models:
    x = compression[model].iloc[:-1]
    y = var_latency_mean[model].iloc[:-1]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Latency [%]')
    plt.scatter(x, y, label=model, alpha=ALPHA)
plt.legend(loc='lower left', frameon=frameon)

plt.figure()
for quantization in quantizations[:-1]:
    x = compression.loc[quantization, :]
    y = var_latency_mean.loc[quantization, :]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Latency [%]')
    plt.scatter(x, y, label=quantization, alpha=ALPHA)
plt.legend(loc='lower left', frameon=FRAMEON)

# Tokens/s x Compression
plt.figure()
for model in models:
    x = compression[model].iloc[:-1]
    y = var_tokens_per_sec[model].iloc[:-1]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Tokens/s [%]')
    plt.scatter(x, y, label=model, alpha=ALPHA)
plt.legend(loc='upper left', frameon=FRAMEON)

plt.figure()
for quantization in quantizations[:-1]:
    x = compression.loc[quantization, :]
    y = var_tokens_per_sec.loc[quantization, :]
    plt.xlabel('Compression [%]')
    plt.ylabel('$\Delta$Tokens/s [%]')
    plt.scatter(x, y, label=quantization, alpha=ALPHA)
plt.legend(loc='upper left', frameon=FRAMEON)