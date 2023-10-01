import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser('CTSI calculation')
parser.add_argument('--bayes_monthly', type=str, help='path of monthly bayesian probabilities differences')
parser.add_argument('--output', type=str, help='output file of CTSI data framework')
month_list = ['2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08',
              '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05',
              '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01',
               '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09',
               '2022-10', '2022-11', '2022-12']
list_mse = []

for m in month_list:
    df = pd.read_csv(args.bayes_monthly+'sort_diff_bayes_%s.csv' % m, header=0)
    percent = df['sort'].quantile(0.95)
    df = df[df['sort'] > percent]
    if len(df) == 0:
        list_mse.append(0)
    else:
        summ = 0
        for j in range(len(df)):
            summ += df.iloc[j]['sort']**2
        list_mse.append(summ/len(df))
print(list_mse)
normal_list_mse = []
for i in list_mse:
    normal_list_mse.append(i-min(list_mse))
normal_list_mse /= max(list_mse)-min(list_mse)
print(normal_list_mse)
df = pd.DataFrame({'CTSI': normal_list_mse}, index=month_list)
df.to_csv(args.output)
plt.plot(month_list, normal_list_mse)
plt.xticks(rotation=90)
plt.show()
