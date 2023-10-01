import logomaker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser('Analyze top amino acid site influencing virus host tropism')
parser.add_argument('--data_adapt', type=str, help='data framework of specific host tropic amino acid probabilities')
parser.add_argument('--data_nonadapt', type=str, help='data framework of other host amino acid probabilities')
parser.add_argument('--sorted_diff', type=str, help='Sorted difference of tropic and other host amino acid probabilities')
parser.add_argument('--out_path', type=str, help='Output path of logo plot figures')

df_adapt = pd.read_csv(args.data_adapt, index_col=0)
df_inadapt = pd.read_csv(args.data_nonadapt, index_col=0)
df_adapt = df_adapt.replace(-1, 0)
df_inadapt = df_inadapt.replace(-1, 0)
df_sort = df_adapt-df_inadapt
sort = []
for i in range(len(df_sort)):
    line = df_sort.iloc[i]
    sort.append(line.max())
df_sort_new = pd.DataFrame({'sort': sort}, index=df_adapt.index)
df_sort_new.sort_values(by='sort', ascending=False).to_csv(args.sorted_diff)
df_sort_new = df_sort_new.sort_values(by='sort', ascending=False)
df_adapt = pd.concat([df_adapt, df_sort_new], axis=1)
df_adapt = df_adapt.sort_values(by='sort', ascending=False)
df_adapt = df_adapt.drop('sort', 1)

df_inadapt = pd.concat([df_inadapt, df_sort_new], axis=1)
df_inadapt = df_inadapt.sort_values(by='sort', ascending=False)
df_inadapt = df_inadapt.drop('sort', 1)
df_inadapt = -df_inadapt

df_adapt = df_adapt.iloc[:20]
df_adapt = df_adapt.replace(-1, 0)
print(df_adapt.index)
x = df_adapt.index.tolist()
x_new = []
for i in x:
    if i > 213:
        x_new.append(i-3)
    else:
        x_new.append(i)
print(x_new)
df_adapt.index = range(20)
df_inadapt = df_inadapt.iloc[:20]
df_inadapt = df_inadapt.replace(1, 0)
df_inadapt.index = range(20)
print(len(df_adapt.columns))

logo = logomaker.Logo(df_adapt, color_scheme='NajafabadiEtAl2017', font_name='Stencil Std', vpad=.1, width=.8)
# logo_inadapt = logomaker.Logo(df_inadapt, color_scheme='NajafabadiEtAl2017', font_name='Stencil Std', vpad=.1, width=.8)
logo.style_spines(visible=False)
logo.style_spines(spines=['left', 'bottom'], visible=True)
logo.style_xticks(rotation=0)
# logo.ax.set_ylabel('Bayesian probability')
# logo.ax.set_xticks(df_adapt.index)
logo.ax.set_xticklabels(x_new)
plt.savefig(args.out_path+'logo_plot_tropic.png', format='png', dpi=600, bbox_inches='tight')

logo_inadapt = logomaker.Logo(df_inadapt, color_scheme='NajafabadiEtAl2017', font_name='Stencil Std', vpad=.1, width=.8)
logo_inadapt.style_spines(visible=False)
logo_inadapt.style_spines(spines=['left', 'top', 'bottom'], visible=True)
logo_inadapt.style_xticks(rotation=0)
# logo_inadapt.ax.set_xticks(df_inadapt.index)
logo_inadapt.ax.set_xticklabels(x_new)
plt.savefig(args.out_path+'logo_plot_non_tropic.png', format='png', dpi=600, bbox_inches='tight')
