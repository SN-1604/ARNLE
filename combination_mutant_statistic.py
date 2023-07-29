import pandas as pd
import itertools
import copy
from tqdm import tqdm
import time
import numpy as np

dic_mutant = {'478': 'K', '452': 'R', '142': 'D', '27': 'S', '25': '-', '24': '-', '26': '-', '376': 'A', '371': 'F',
              '405': 'N', '19': 'I', '408': 'S', '501': 'Y', '213': 'G', '681': 'H', '156': 'G', '157': '-', '158': '-',
              '950': 'N', '69': '-'}
dic_origin = {'478': 'T', '452': 'L', '142': 'G', '27': 'A', '25': 'P', '24': 'L', '26': 'P', '376': 'T', '371': 'S',
              '405': 'D', '19': 'T', '408': 'R', '501': 'N', '213': 'V', '681': 'P', '156': 'E', '157': 'F', '158': 'R',
              '950': 'D', '69': 'H'}
dic_extra = {'19': 'R', '681': 'R'}
df_amino = pd.read_csv('df_amino_add_host_revised.csv', index_col=0)
df_amino['index'] = range(len(df_amino))
sites = dic_mutant.keys()
dic_bayes = {}
dic_combination = {}
# for i in range(11, 12):
for i in range(1, len(dic_mutant.keys())+1):
    tmp = list(itertools.combinations(dic_mutant.keys(), i))
    # print(tmp)
    print('length', i)
    with tqdm(total=len(tmp)) as pbar:
        for j in tmp:
            dic_tmp_mutant = {}
            dic_tmp_origin = {}
            name = ''
            for s in j:
                if int(s) > 213:
                    key = str(int(s)+3)
                else:
                    key = s
                dic_tmp_mutant[key] = list(dic_mutant[s])
                if dic_mutant[s] == '-':
                    name += dic_origin[s] + s + 'del+'
                else:
                    name += dic_origin[s]+s+dic_mutant[s]+'+'
            name = name.rstrip('+')
            df_tmp_mutant = df_amino[list(dic_tmp_mutant.keys())]
            # start = time.time()
            index_tmp_mutant = df_tmp_mutant.isin(dic_tmp_mutant)[list(dic_tmp_mutant.keys())].all(1)
            print(df_amino['index'][index_tmp_mutant].tolist())
            df_prob = df_amino['Prob_PR'][index_tmp_mutant]
            amount1 = len(df_prob[df_prob >= 0.5])
            amount0 = len(df_prob[df_prob < 0.5])
            # amount1 = len(df_amino[index_tmp_mutant][df_amino['Prob_PR'] >= 0.5])
            # amount0 = len(df_amino[index_tmp_mutant][df_amino['Prob_PR'] < 0.5])
            # print(time.time()-start)

            # print(amount1, dic_tmp_mutant)
            if len(index_tmp_mutant) > 50:
                dic_bayes[name] = (amount1-amount0)/len(index_tmp_mutant)
            else:
                dic_bayes[name] = 0
            # if amount1 == 27788 and amount0 == 1988:
            #     print(name)
            #     print(len(index_tmp_mutant))
            #     print(dic_bayes['T478K+G142D+A27S+T376A+S371F+D405N+T19I+R408S+N501Y+V213G+P681H'])

            for e in dic_extra.keys():
                if int(e) > 213:
                    key = str(int(e) + 3)
                else:
                    key = e
                if key in dic_tmp_mutant.keys():
                    dic_tmp_mutant_new = copy.deepcopy(dic_tmp_mutant)
                    dic_tmp_mutant_new[key] = list(dic_extra[e])
                    index_name = name.index(e)
                    name_list = list(name)
                    name_list[index_name+len(e)] = dic_extra[e]
                    name_new = ''.join(name_list)
                    df_tmp_mutant_new = df_amino[list(dic_tmp_mutant_new.keys())]
                    index_tmp_mutant_new = df_tmp_mutant_new.isin(dic_tmp_mutant_new)[list(dic_tmp_mutant_new.keys())].all(1)
                    df_prob_new = df_amino['Prob_PR'][index_tmp_mutant_new]
                    # amount1_new = len(df_amino.loc[index_tmp_mutant_new][df_amino['Prob_PR'] >= 0.5])
                    # amount0_new = len(df_amino.loc[index_tmp_mutant_new][df_amino['Prob_PR'] < 0.5])
                    amount1_new = len(df_prob_new[df_prob_new >= 0.5])
                    amount0_new = len(df_prob_new[df_prob_new < 0.5])
                    if len(index_tmp_mutant_new) > 50:
                        dic_bayes[name_new] = (amount1_new - amount0_new) / len(index_tmp_mutant_new)
                    else:
                        dic_bayes[name_new] = 0

            if '19' in dic_tmp_mutant.keys() and '684' in dic_tmp_mutant.keys():
                dic_tmp_mutant_new = copy.deepcopy(dic_tmp_mutant)
                dic_tmp_mutant_new['19'] = list(dic_extra['19'])
                dic_tmp_mutant_new['684'] = list(dic_extra['681'])
                name_list = list(name)
                name_list[name.index('19')+2] = dic_extra['19']
                name_list[name.index('681')+3] = dic_extra['681']
                name_new = ''.join(name_list)
                df_tmp_mutant_new = df_amino[list(dic_tmp_mutant_new.keys())]
                index_tmp_mutant_new = df_tmp_mutant_new.isin(dic_tmp_mutant_new)[list(dic_tmp_mutant_new.keys())].all(1)
                df_prob_new = df_amino['Prob_PR'][index_tmp_mutant_new]
                # amount1_new = len(df_amino.loc[index_tmp_mutant_new][df_amino['Prob_PR'] >= 0.5])
                # amount0_new = len(df_amino.loc[index_tmp_mutant_new][df_amino['Prob_PR'] < 0.5])
                amount1_new = len(df_prob_new[df_prob_new >= 0.5])
                amount0_new = len(df_prob_new[df_prob_new < 0.5])
                if len(index_tmp_mutant_new) > 50:
                    dic_bayes[name_new] = (amount1_new - amount0_new) / len(index_tmp_mutant_new)
                else:
                    dic_bayes[name_new] = 0

            # df_tmp_origin = df_amino[list(dic_tmp_origin.keys())]
            # index_tmp_origin = df_tmp_origin[df_tmp_origin.isin(dic_tmp_origin).all(1)].index
            # amount0 = len(df_amino.loc[index_tmp_origin])
            pbar.update(1)
# print(sorted(dic_bayes.items(), key=lambda x: x[1], reverse=True))
print(dic_bayes['T478K+G142D+A27S+T376A+S371F+D405N+T19I+R408S+N501Y+V213G+P681H'])
# np.save('dic_combination_bates_add_host.npy', dic_bayes)
# df_new = pd.DataFrame(dic_bayes, index=['bayes']).T
# print(df_new)
# df_new.to_excel('combination_bayes_add_host_revised.xlsx')
# df_new.to_excel('combination_test.xlsx')
