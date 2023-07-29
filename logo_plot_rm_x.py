import logomaker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df_adapt = pd.read_csv('bayes_test_total_adapt.csv', index_col=0)
# df_inadapt = pd.read_csv('bayes_test_total_nonadapt.csv', index_col=0)
# df_adapt = pd.read_csv('bayes_trying_new_revised_adapt_new_add_duplicated.csv', index_col=0)
# df_inadapt = pd.read_csv('bayes_trying_new_revised_nonadapt_new_add_duplicated.csv', index_col=0)
# df_adapt = pd.read_csv('bayes_trying_new_revised_adapt_new_add_duplicated.csv', index_col=0)
# df_inadapt = pd.read_csv('bayes_trying_new_revised_nonadapt_new_add_duplicated.csv', index_col=0)
# df_adapt = pd.read_csv('bayes_trying_new_add_BA_only_complete_total_revised_adapt.csv', index_col=0)
# df_inadapt = pd.read_csv('bayes_trying_new_add_BA_only_complete_total_revised_nonadapt.csv', index_col=0)
df_adapt = pd.read_csv('bayes_add_host_revised_adapt.csv', index_col=0)
df_inadapt = pd.read_csv('bayes_add_host_revised_nonadapt.csv', index_col=0)
df_adapt = df_adapt.replace(-1, 0)
df_inadapt = df_inadapt.replace(-1, 0)
df_sort = df_adapt-df_inadapt
sort = []
for i in range(len(df_sort)):
    line = df_sort.iloc[i]
    sort.append(line.max())
df_sort_new = pd.DataFrame({'sort': sort}, index=df_adapt.index)
# df_sort_new.sort_values(by='sort', ascending=False).to_csv('sort_diff_bayes_india_rm_x.csv')
# df_sort_new.sort_values(by='sort', ascending=False).to_csv('sort_diff_bayes_test_total.csv')
# df_sort_new.sort_values(by='sort', ascending=False).to_csv('sort_diff_bayes_trying_new_revised_new_add_duplicated.csv')
# df_sort_new.sort_values(by='sort', ascending=False).to_csv('sort_diff_bayes_trying_new_add_BA_only_complete_total_revised.csv')
df_sort_new.sort_values(by='sort', ascending=False).to_csv('sort_diff_bayes_add_host_revised.csv')
df_sort_new = df_sort_new.sort_values(by='sort', ascending=False)
# print(df_sort_new)
df_adapt = pd.concat([df_adapt, df_sort_new], axis=1)
df_adapt = df_adapt.sort_values(by='sort', ascending=False)
df_adapt = df_adapt.drop('sort', 1)
# df_adapt.to_csv('sort_bayes_0411_adapt_india_rm_x.csv')
# df_adapt.to_csv('sort_bayes_adapt_india_delta_merged_rm_x_revised.csv')
# df_adapt.to_csv('sort_bayes_adapt_trying_new_revised.csv')

df_inadapt = pd.concat([df_inadapt, df_sort_new], axis=1)
df_inadapt = df_inadapt.sort_values(by='sort', ascending=False)
df_inadapt = df_inadapt.drop('sort', 1)
# df_inadapt.to_csv('sort_bayes_0411_inadapt_india_rm_x.csv')
# df_inadapt.to_csv('sort_bayes_nonadapt_india_delta_merged_rm_x_revised.csv')
# df_adapt.to_csv('sort_bayes_nonadapt_trying_new_revised.csv')
df_inadapt = -df_inadapt

df_adapt = df_adapt.iloc[:20]
df_adapt = df_adapt.replace(-1, 0)
print(df_adapt.index)
x = df_adapt.index.tolist()
# print(x)
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
# plt.xticks(x_new)
# plt.show()
# plt.savefig('./figure/logo_plot_adapt_trying_new_revised_new_add_duplicated.png', format='png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure/logo_plot_adapt_bayes_minus_20_india.png', format='png', dpi=1200)
# plt.savefig('./figure/logo_plot_adapt_trying_new_add_BA_only_complete_total_revised.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig('./figure/logo_plot_adapt_add_host_revised.png', format='png', dpi=600, bbox_inches='tight')

logo_inadapt = logomaker.Logo(df_inadapt, color_scheme='NajafabadiEtAl2017', font_name='Stencil Std', vpad=.1, width=.8)
logo_inadapt.style_spines(visible=False)
logo_inadapt.style_spines(spines=['left', 'top', 'bottom'], visible=True)
logo_inadapt.style_xticks(rotation=0)
# logo_inadapt.ax.set_xticks(df_inadapt.index)
logo_inadapt.ax.set_xticklabels(x_new)
# logo_inadapt.ax.set_xlabel('Amino acid site')
# plt.show()
# plt.savefig('./figure/logo_plot_nonadapt_trying_new_revised_new_add_duplicated_revised.png', format='png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure/logo_plot_nonadapt_trying_new_add_BA_only_complete_total_revised.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig('./figure/logo_plot_nonadapt_add_host_revised.png', format='png', dpi=600, bbox_inches='tight')
