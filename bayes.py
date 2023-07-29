import numpy as np
import pandas as pd

# df = pd.read_csv('df_amino_trying_new_revised_new_add_duplicated.csv', index_col=0, engine='python')
# df = pd.read_csv('df_amino_trying_new_add_BA.csv', index_col=0, engine='python')
# df = pd.read_csv('df_amino_trying_new_add_BA_only_complete_total_revised.csv', index_col=0, engine='python')
# df = pd.read_csv('df_amino_trying_new_add_BA_only_complete_total_revised_add_host_XBB.csv', index_col=0, engine='python')
df = pd.read_csv('df_amino_add_host_revised.csv', index_col=0, engine='python')
df = df.fillna('-')
amino_total = set()
# print(df.columns[2:-5].tolist())
# for i in df.columns[2:-5]:
print(df.columns[2:-6].tolist())
for i in df.columns[2:-6]:
    # print(df[i])
    amino_total.update(set(df[i]))
# amino_total.remove('X')
# print(amino_total)
amino_total = sorted(amino_total)
amount_total = len(df)
amount_adapt = len(df[df['Prob_PR'] >= 0.5])
amount_inadapt = len(df[df['Prob_PR'] < 0.5])
print(amount_adapt)
print(amount_inadapt)

dic = {}
dic_adapt = {}
dic_inadapt = {}
# for i in df.columns[1:-6]:
# for i in df.columns[2:-5]:
for i in df.columns[2:-6]:
    print(i)
    amino_set = set(df[i])
    for j in amino_total:
        if j in amino_set:
            amount1 = len(df[df[i] == j][df['Prob_PR'] >= 0.5])
            amount0 = len(df[df[i] == j][df['Prob_PR'] < 0.5])

            # if amount0 != 0:
            #     if amount1 >= 50 and amount0 >= 50:
            #         # prob = (amount_inadapt*amount1)/(amount_adapt*amount0)
            #         prob = (amount1/amount_adapt)-(amount0/amount_inadapt)
            #     else:
            #         prob = -2
            # else:
            #     if amount1 >= 50:
            #         prob = -1
            #     else:
            #         prob = -2
            # if j not in dic.keys():
            #     dic[j] = [prob]
            # else:
            #     dic[j].append(prob)

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
            # if j not in dic.keys():
            #     dic[j] = [0]
            # else:
            #     dic[j].append(0)

            if j not in dic_adapt.keys():
                dic_adapt[j] = [0]
            else:
                dic_adapt[j].append(0)
            if j not in dic_inadapt.keys():
                dic_inadapt[j] = [0]
            else:
                dic_inadapt[j].append(0)
# df_new = pd.DataFrame(dic, index=df.columns[1:-6].tolist())
# # df_new.to_csv('bayes_0411.csv')
# # df_new.to_csv('bayes_0411_minus_revised.csv')
# # df_new.to_csv('bayes_lambda_rm_x.csv')
# # df_new.to_csv('bayes_delta_sampled.csv')
# df_new.to_csv('bayes_india_delta_merged_rm_x_revised.csv')

# df_adapt = pd.DataFrame(dic_adapt, index=df.columns[2:-5].tolist())
# df_inadapt = pd.DataFrame(dic_inadapt, index=df.columns[2:-5].tolist())
df_adapt = pd.DataFrame(dic_adapt, index=df.columns[2:-6].tolist())
df_inadapt = pd.DataFrame(dic_inadapt, index=df.columns[2:-6].tolist())
# df_adapt = pd.DataFrame(dic_adapt, index=df.columns[1:-6].tolist())
# df_inadapt = pd.DataFrame(dic_inadapt, index=df.columns[1:-6].tolist())
# df_adapt.to_csv('bayes_trying_new_revised_adapt_new_add_duplicated.csv')
# df_inadapt.to_csv('bayes_trying_new_revised_nonadapt_new_add_duplicated.csv')
# df_adapt.to_csv('bayes_trying_new_add_BA_adapt.csv')
# df_inadapt.to_csv('bayes_trying_new_add_BA_nonadapt.csv')
# df_adapt.to_csv('bayes_trying_new_add_BA_only_complete_total_revised_adapt.csv')
# df_inadapt.to_csv('bayes_trying_new_add_BA_only_complete_total_revised_nonadapt.csv')
# df_adapt.to_csv('bayes_trying_new_add_BA_only_complete_remove_BA.1_adapt_revised.csv')
# df_inadapt.to_csv('bayes_trying_new_add_BA_only_complete_remove_BA.1_nonadapt_revised.csv')
# df_adapt.to_csv('bayes_trying_new_add_BA_only_complete_total_revised_add_host_XBB_adapt.csv')
# df_inadapt.to_csv('bayes_trying_new_add_BA_only_complete_total_revised_add_host_XBB_nonadapt.csv')
df_adapt.to_csv('bayes_add_host_revised_adapt.csv')
df_inadapt.to_csv('bayes_add_host_revised_nonadapt.csv')

# df_adapt = pd.read_csv('bayes_india_delta_merged_adapt_rm_x.csv')
# df_inadapt = pd.read_csv('bayes_india_delta_merged_nonadapt_rm_x.csv')
# df = df_adapt-df_inadapt
# df_max = df.max(axis=1)
# df_new = df_max.sort_values(ascending=False)
# print(df_new)

# # df = pd.read_csv('bayes_0411.csv', index_col=0)
# # df = pd.read_csv('bayes_0411_minus_revised.csv', index_col=0)
# # df = pd.read_csv('bayes_lambda_rm_x.csv', index_col=0)
# # df = pd.read_csv('bayes_delta_sampled.csv', index_col=0)
# df = pd.read_csv('bayes_india_delta_merged_rm_x.csv', index_col=0)
# df_max = df.max(axis=1)
# df_new = df_max.sort_values(ascending=False)
# # df_new.to_csv('bayes_0411_sorted.csv')
# # df_new.to_csv('bayes_lambda_rm_x_sorted.csv')
# # df_new.to_csv('bayes_delta_sampled_sorted.csv')
# print(df_max.sort_values(ascending=False))
