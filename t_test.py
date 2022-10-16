import numpy as np
import pandas as pd
from scipy import stats
import os
from os.path import exists

dir_path = os.path.dirname(os.path.realpath(__file__))
runs = 10
generations = 30


# outfile = pd.read_csv()

def t_test(generations, experiment_name_1, experiment_name_2, csv):
    file_path_1 = os.path.join(dir_path, experiment_name_1, csv)
    file_path_2 = os.path.join(dir_path, experiment_name_2, csv)

    # print(file_path_1)
    df_1 = pd.read_csv(file_path_1)
    df_2 = pd.read_csv(file_path_2)

    mean_per_gen_1 = np.zeros(generations)
    mean_per_gen_2 = np.zeros(generations)
    for gen in range(generations):
        mean_gain_1 = (df_1['mean'].where(df_1['gen']==gen)).to_numpy()
        mean_gain_2 = (df_2['mean'].where(df_2['gen']==gen)).to_numpy()
        mean_per_gen_1[gen] = np.mean(mean_gain_1[~np.isnan(mean_gain_1)])
        mean_per_gen_2[gen] = np.mean(mean_gain_2[~np.isnan(mean_gain_2)])
    t_test = stats.ttest_ind(mean_per_gen_1, mean_per_gen_2)
    return t_test

t_test_engroup1 = t_test(generations, 'results_time_engroup1', 'results_selfadapt_engroup1', 'results.csv')
t_test_engroup2 = t_test(generations, 'results_selfadapt_engroup2', 'results_time_engroup2', 'results.csv')
# t_test_enemy_7 = t_test(generations, 'steadystate_7', 'generational_7', 'results.csv')

print("enemy group 1 t-test:", t_test_engroup1)
print("enemy group 2 t-test:", t_test_engroup2)
