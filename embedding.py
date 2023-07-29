from elmoformanylangs import Embedder
from gensim.models import word2vec
import re
import numpy as np
import math
import copy
from tqdm import tqdm

protein = 'S'
split = 6
batchsize = 32
elmo = Embedder('G:/Py Files/ELMoForManyLangs-master/output/all')

dic_host = {'human': 0, 'bat': 1, 'carnivora': 2, 'artiodactyla': 3, 'swine': 4}
dic_f1 = {'ORF1ab': 0.86, 'E': 0.884, 'M': 0.887, 'N': 0.93}

# f = word2vec.LineSentence('S_test_0411_deleted_rm_x.fasta')
# f = word2vec.LineSentence('S_test_after_0801_sampled.fasta')
# f = word2vec.LineSentence('S_for_trying.fasta')
# f = word2vec.LineSentence('BA.4.fasta')
# f = word2vec.LineSentence('S_test_total.fasta')
# f = word2vec.LineSentence('BA.1-5_new.fasta')
# f = word2vec.LineSentence('data_test_for_align_unaligned.fasta')
# f = word2vec.LineSentence('BQ.1_new_new.fasta')
# f = word2vec.LineSentence('BA.5_new_new.fasta')
# f = word2vec.LineSentence('BA.1-5_only_complete_total.fasta')
f = word2vec.LineSentence('nonhuman_host_variant.fasta')
t = 1
sequence = []
label = []
for i in f:
    if t % 2 == 0:
        sequence.append(i[0].rstrip('*'))
    else:
        label.append(i[0].lstrip('>'))
    t += 1
reviews = []
for i in sequence:
    reviews.append(re.findall('.{%d}' % split, i))
print(len(reviews))

tmp = []
for i in reviews:
    tmp.append(len(i))
print(max(tmp))
sequence_length = max(tmp)
reviews = np.asarray(reviews)
label = np.asarray(label)
# print(label.shape)
# reviews = reviews[67552:]
print(reviews.shape)

# np.random.seed(100)
# seed = np.random.choice(np.arange(0, reviews.shape[0]), 30000, replace=False)
# reviews = reviews[seed]
# label = label[seed]
# np.save('label_sampled_BA.2.npy', label)
# np.save('reviews_sampled_BA.2.npy', reviews)

sequence_length = 264

# print(x_train.shape[0])
data = []
with tqdm(total=math.ceil(reviews.shape[0]/batchsize)) as pbar:
    for b in range(math.ceil(reviews.shape[0]/batchsize)):
        for i in elmo.sents2elmo(reviews[b*batchsize:(b+1)*batchsize]):
            origin = copy.deepcopy(i)
            origin = origin.astype(np.float32)
            while origin.shape[0] < sequence_length:
                origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
            data.append(origin)
        pbar.update(1)
data = np.asarray(data)
print(data.shape)
# np.save('data_test_SARS_MERS_adjusted_before_0411.npy', data)
# np.save('data_test_after_0801.npy', data)
# np.save('data_test_BA.4.npy', data)
# np.save('J:/coronavirus_final_revise/data_test_BA_new.npy', data)
# np.save('F:/coronavirus_final_revise/data_test_before_omicron_1.npy', data)
# np.save('data_test_BF.7.npy', data)
# np.save('data_test_omicron.npy', data)
# np.save('data_test_BA.5_new_new.npy', data)
# np.save('data_test_BA.1-5_only_complete_total.npy', data)
# np.save('data_train_for_trying.fasta', data)
np.save('data_test_nonhuman_host_variant.npy', data)

# for t in range(8):
#     f = open('./test_after_0411/S_test_after_0411_sampled_%d_rm_x.fasta' % (t+1)).read().splitlines()
#     sequence_test = []
#     data = []
#     for i in range(1, len(f), 2):
#         sequence_test.append(f[i])
#     reviews_test = []
#     length_test = []
#     for i in sequence_test:
#         reviews_test.append(re.findall('.{%d}' % split, i))
#         length_test.append(len(re.findall('.{%d}' % split, i)))
#     del f
#
#     data_test = []
#     tick = 1
#     with tqdm(total=math.ceil(len(reviews_test)/batchsize)) as pbar:
#         for b in range(math.ceil(len(reviews_test)/batchsize)):
#             for i in elmo.sents2elmo(reviews_test[b*batchsize:(b+1)*batchsize]):
#                 origin = copy.deepcopy(i)
#                 origin = origin.astype(np.float32)
#                 while origin.shape[0] < sequence_length:
#                     origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
#                 data_test.append(origin)
#             pbar.update(1)
#     data_test = np.asarray(data_test)
#     print(data_test.shape)
#     # np.save('G:/Py Files/coronavirus/explore/data_test_%s_elmo_extra_new.npy' % protein, data_test)
#     np.save('data_test_S_SARS_MERS_adjusted_after_0411_%d.npy' % (t+1), data_test)

# # x_train = np.load('x_train_S_SARS_MERS_adjusted_revised.npy', allow_pickle=True)
# # x_val = np.load('x_val_S_SARS_MERS_adjusted_revised.npy', allow_pickle=True)
# # x_train = np.load('x_train_S_SARS_MERS_human_revised_rm_except_plus.npy', allow_pickle=True)
# # x_val = np.load('x_val_S_SARS_MERS_human_revised_rm_except_plus.npy', allow_pickle=True)
# # x_train = np.load('x_train_S_trying_new_revised_except.npy', allow_pickle=True)
# # x_val = np.load('x_val_S_trying_new_revised_except.npy', allow_pickle=True)
# # x_train = np.load('x_train_S_trying_new_revised_except.npy', allow_pickle=True)
# # x_val = np.load('x_val_S_trying_new_revised_except.npy', allow_pickle=True)
# x_train = np.load('x_train_S_for_trying.npy', allow_pickle=True)
# x_val = np.load('x_val_S_for_trying.npy', allow_pickle=True)
# print(x_train.shape[0])
# data = []
# for b in range(math.ceil(x_train.shape[0]/batchsize)):
#     for i in elmo.sents2elmo(x_train[b*batchsize:(b+1)*batchsize]):
#         origin = copy.deepcopy(i)
#         origin = origin.astype(np.float32)
#         while origin.shape[0] < sequence_length:
#             origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
#         data.append(origin)
# data = np.asarray(data)
# print(data.shape)
# # np.save('data_train_SARS_MERS_adjusted_revised.npy', data)
# # np.save('data_train_SARS_MERS_human_revised_rm_except_plus.npy', data)
# # np.save('data_train_trying_new_revised_except.npy', data)
# np.save('data_train_for_trying.npy', data)
#
# data = []
# for b in range(math.ceil(x_val.shape[0]/batchsize)):
#     for i in elmo.sents2elmo(x_val[b*batchsize:(b+1)*batchsize]):
#         origin = copy.deepcopy(i)
#         origin = origin.astype(np.float32)
#         while origin.shape[0] < sequence_length:
#             origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
#         data.append(origin)
# data = np.asarray(data)
# print(data.shape)
# # np.save('data_val_SARS_MERS_adjusted_revised.npy', data)
# # np.save('data_val_SARS_MERS_human_revised_rm_except_plus.npy', data)
# np.save('data_val_for_trying.npy', data)

# f_test = word2vec.LineSentence('G:/Py Files/coronavirus/explore/variants/omicron_sampled_rm_x.fasta')
# t = 1
# sequence_test = []
# label = []
# for i in f_test:
#     if t % 2 == 0:
#         sequence_test.append(i[0])
#     else:
#         label.append(i[0].lstrip('>'))
#     t += 1
# reviews_test = []
# for i in sequence_test:
#     reviews_test.append(re.findall('.{%d}' % split, i))
# label_new = []
# f_new = open('omicron_sampled_rm_x_aligned_revised.fasta').read().splitlines()
# for i in f_new:
#     if '>' in i:
#         label_new.append(i.lstrip('>').replace(' ', '_'))
# filt = []
# for i in range(len(label)):
#     if label[i] in label_new:
#         filt.append(i)
# print(filt)
# reviews_test = np.asarray(reviews_test)
# reviews_test = reviews_test[filt]
# # print(reviews_test)
# data = []
# with tqdm(total=math.ceil(reviews_test.shape[0]/batchsize)) as pbar:
#     for b in range(math.ceil(reviews_test.shape[0]/batchsize)):
#         for i in elmo.sents2elmo(reviews_test[b*batchsize:(b+1)*batchsize]):
#             origin = copy.deepcopy(i)
#             origin = origin.astype(np.float32)
#             while origin.shape[0] < sequence_length:
#                 origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
#             data.append(origin)
#         pbar.update(1)
# data = np.asarray(data)
# print(data.shape)
# np.save('data_omicron_sampled_rm_x_align_revised.npy', data)
