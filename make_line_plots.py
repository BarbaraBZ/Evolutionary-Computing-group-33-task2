import numpy as np
import pandas as pd
import os
from os.path import exists
from matplotlib import pyplot as plt

experiment_name1 = "results_time_engroup1"
experiment_name2 = "results_selfadapt_engroup1"
file_name_csv = "results.csv"
runs = 10
generations = 30
enemy = 6


def txt_to_csv(experiment_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, experiment_name)
    if not exists(os.path.join(file_path, file_name_csv)):
        print("Writing results to \"" + file_name_csv + "\"")
        results = pd.read_csv(os.path.join(file_path, "results.txt"), error_bad_lines=False)
        results.to_csv(os.path.join(file_path, file_name_csv), index=None)
    else:
        print("Getting results from existing file \"" + file_name_csv + "\"")
        results = pd.read_csv(os.path.join(file_path, file_name_csv))
    return results

def line_plot(runs, generations, experiment_name):
    df = txt_to_csv(experiment_name)
    mean_per_gen = np.zeros(generations)
    var_per_gen = np.zeros(generations)
    max_per_gen = np.zeros(generations)
    var_max_per_gen = np.zeros(generations)
    for gen in range(generations):
        mean_gen = (df['mean'].where(df['gen']==gen)).to_numpy()
        mean_per_gen[gen] = np.mean(mean_gen[~np.isnan(mean_gen)])
        var_gen = (df['std']).where(df['gen']==gen).to_numpy()
        var_per_gen[gen] = np.mean(var_gen[~np.isnan(var_gen)])
        max_gen = (df['best'].where(df['gen']==gen)).to_numpy()
        max_per_gen[gen] = np.mean(max_gen[~np.isnan(max_gen)])
        var_max_per_gen[gen] = np.std(max_gen[~np.isnan(max_gen)])
    return mean_per_gen, max_per_gen, var_per_gen, var_max_per_gen


means_time_en1, maxs_time_en1, var_means_time_en1, var_maxs_time_en1 = line_plot(runs, generations, experiment_name1)
means_adapt_en1, maxs_adapt_en1, var_means_adapt_en1, var_maxs_adapt_en1 = line_plot(runs, generations, experiment_name2)



x = np.linspace(0, generations, generations)
plt.plot(x, means_time_en1, label = "mean fitness time dependent")
plt.plot(x, maxs_time_en1, label = "max fitness time dependent")

plt.plot(x, means_adapt_en1, label = "mean fitness self adaptive")
plt.plot(x, maxs_adapt_en1, label = "max fitness self adaptive")

plt.fill_between(x, means_time_en1-var_means_time_en1, means_time_en1+var_means_time_en1, alpha = 0.1)
plt.fill_between(x, maxs_time_en1-var_maxs_time_en1, maxs_time_en1+var_maxs_time_en1, alpha = 0.1)

plt.fill_between(x, means_adapt_en1-var_means_adapt_en1, means_adapt_en1+var_means_adapt_en1, alpha = 0.1)
plt.fill_between(x, maxs_adapt_en1-var_maxs_adapt_en1, maxs_adapt_en1+var_maxs_adapt_en1, alpha = 0.1)

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness with std of time dependent and self adaptive against enemy group 1")
plt.legend()
plt.show()
