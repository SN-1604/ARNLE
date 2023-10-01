import numpy as np
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser('Perform Post-hoc Bayesian explanation')
parser.add_argument('--data_frame', type=str, help='pandas data framework of merged sequences and predicting probabilities')
parser.add_argument('--data_adapt', type=str, help='output data framework of specific host tropic amino acid probabilities')
parser.add_argument('--data_nonadapt', type=str, help='output data framework of other host amino acid probabilities')

df = pd.read_csv(args.data_frame, index_col=0, engine='python')
df = df.fillna('-')
amino_total = set()
print(df.columns[2:-6].tolist())
for i in df.columns[2:-6]:
    amino_total.update(set(df[i]))
amino_total = sorted(amino_total)
amount_total = len(df)
amount_adapt = len(df[df['Prob_PR'] >= 0.5])
amount_inadapt = len(df[df['Prob_PR'] < 0.5])
print(amount_adapt)
print(amount_inadapt)

dic = {}
dic_adapt = {}
dic_inadapt = {}
for i in df.columns[2:-6]:
    print(i)
    amino_set = set(df[i])
    for j in amino_total:
        if j in amino_set:
            amount1 = len(df[df[i] == j][df['Prob_PR'] >= 0.5])
            amount0 = len(df[df[i] == j][df['Prob_PR'] < 0.5])

            if amount1 > 50:
                prob_adapt = amount1/amount_adapt
            else:
                prob_adapt = -1
            if amount0 > 50:
                prob_inadapt = amount0/amount_inadapt
            else:
                prob_inadapt = -1
            if j not in dic_adapt.keys():
                dic_adapt[j] = [prob_adapt]
            else:
                dic_adapt[j].append(prob_adapt)
            if j not in dic_inadapt.keys():
                dic_inadapt[j] = [prob_inadapt]
            else:
                dic_inadapt[j].append(prob_inadapt)
        else:

            if j not in dic_adapt.keys():
                dic_adapt[j] = [0]
            else:
                dic_adapt[j].append(0)
            if j not in dic_inadapt.keys():
                dic_inadapt[j] = [0]
            else:
                dic_inadapt[j].append(0)

df_adapt = pd.DataFrame(dic_adapt, index=df.columns[2:-6].tolist())
df_inadapt = pd.DataFrame(dic_inadapt, index=df.columns[2:-6].tolist())
df_adapt.to_csv(args.data_adapt)
df_inadapt.to_csv(args.data_nonadapt)
