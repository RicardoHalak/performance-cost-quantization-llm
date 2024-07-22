"""
In this script I plot the cost over time for on-premise and in the cloud implementation, 
as well as using different models of API providers.

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

FIGSIZE=(12,8)
SAVE_IMAGES_DIR = r'SAVE_DIR'
#%%
costs_dev = {
    'gpt_35': 6000,
    'gpt_4': 6000,
    'claude_3': 6000,
    'mistral_7b': 6000,
    'llama3_8b': 6000,
    'on_prem': 60009.68,
    'cloud': 19430.40
}

monthly_costs_prod = {
    'gpt_35': 4019.44,
    'gpt_4': 45936.45,
    'claude_3': 20671.4,
    'mistral_7b': 528.27,
    'llama3_8b': 1240.28,
    'on_prem': 29.06,
    'cloud': 2568.96
}

five_months_costs_main = {
    'gpt_35': 750,
    'gpt_4': 750,
    'claude_3': 750,
    'mistral_7b': 750,
    'llama3_8b': 750,
    'on_prem': 1500,
    'cloud': 750
}

labels = ["GPT-3.5 Turbo Instruct",
          "GPT-4 Turbo",
          "AWS Claude 3 Sonnet",
          "AWS Mistral 7B",
          "AWS Llama3 8B Instruct",
          "On-premise",
          "Cloud"]

series = {}
n_months = 36
for key in costs_dev.keys():
    series[key] = np.array([costs_dev[key] for i in range(0, n_months + 1)])
    monthly_cost_prod_cumsum = np.array([0 if i in [0, 1] else monthly_costs_prod[key] * (i-2) for i in range(0, n_months + 1)])
    monthly_cost_main_cumsum = np.array([0, 0] + [five_months_costs_main[key] * (i // 5) for i in range(0, n_months - 1)])
    series[key] = series[key] + monthly_cost_prod_cumsum + monthly_cost_main_cumsum

series = pd.DataFrame(series)

ax = series.plot(figsize=FIGSIZE, 
            logy=True, 
            linewidth=2.5)
ax.set_xlabel("Month") 
ax.set_ylabel("Log Total Expenditures [€]")
ax.legend(labels=labels, 
          loc='center right',
          ncols=1,
          frameon=False,
          bbox_to_anchor=(1.75, 0.5)
          )
# plt.legend(loc='lower left', 
#             frameon=FRAMEON, 
#             ncols=4, 
#             bbox_to_anchor=(-0.1, -.5),
#             )    
plt.xlim(0, n_months)
plt.xticks(np.array([*range(0, n_months + 1, 4)]))

save_title = "cost_over_time"
plt.savefig(f'{SAVE_IMAGES_DIR}/{save_title}.svg', 
            format='svg',
            bbox_inches='tight')
