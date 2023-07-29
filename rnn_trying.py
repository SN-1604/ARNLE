import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
import re
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import copy
import math
import random
import copy
import time
from sklearn import metrics
import matplotlib.pyplot as plt

lr = 1e-3
epoch = 30
batchsize = 512
hidden_size = [256, 128]
keepprob = 0.8
sequence_length = 264
tf.disable_v2_behavior()
tf.disable_eager_execution()

protein = 'S'
split = 6

# x_train = np.load('x_train_revised_add_host.npy', allow_pickle=True)
# x_val = np.load('x_val_revised_add_host.npy', allow_pickle=True)
# y_train = np.load('y_train_revised_add_host.npy', allow_pickle=True)
# y_val = np.load('y_val_revised_add_host.npy', allow_pickle=True)
# x_train = np.load('x_train_S_trying_new_revised.npy', allow_pickle=True)
# x_val = np.load('x_val_S_trying_new_revised.npy', allow_pickle=True)
# y_train = np.load('y_train_S_trying_new_revised.npy', allow_pickle=True)
# y_val = np.load('y_val_S_trying_new_revised.npy', allow_pickle=True)
x_train = np.concatenate([np.load('x_train_S_trying_new_revised.npy', allow_pickle=True), np.load('x_train_rodentia.npy', allow_pickle=True)], axis=0)
x_val = np.concatenate([np.load('x_val_S_trying_new_revised.npy', allow_pickle=True), np.load('x_val_rodentia.npy', allow_pickle=True)], axis=0)
y_train = np.concatenate([np.load('y_train_S_trying_new_revised.npy', allow_pickle=True), np.load('y_train_rodentia.npy', allow_pickle=True)], axis=0)
y_val = np.concatenate([np.load('y_val_S_trying_new_revised.npy', allow_pickle=True), np.load('y_val_rodentia.npy', allow_pickle=True)], axis=0)
# print(y_train)
# data_train = np.load('data_train_revised_add_host_smote_6.npy')
data_train = np.load('data_train_revised_add_host.npy')
data_val = np.load('data_val_revised_add_host.npy')
# data_train = np.load('data_train_trying_new_revised.npy')
# data_val = np.load('data_val_trying_new_revised.npy')
print(data_train.shape)
print(data_val.shape)
# y_train = open('label_train_revised_add_host_smote_6.txt').readlines()

dic_host = {'human': 0, 'bat': 1, 'carnivora': 2, 'artiodactyla': 3, 'swine': 4, 'rodentia': 5}
# dic_host = {'Primates': 0, 'Chiroptera': 1, 'Carnivora': 2, 'Artiodactyla': 3, 'Suiformes': 4, 'Rodentia': 5,
#             'Perissodactyla': 6, 'Erinaceomorpha': 7}
label_one_train = []
label_one_val = []
# for i in y_train:
#     label_one_train.append(dic_host[i.rstrip('\n')])
for i in y_train:
    label_one_train.append(dic_host[i.split('_')[0].lower()])
for i in y_val:
    if len(i.split('_')) == 2:
        label_one_val.append(dic_host[i.split('_')[0].lower()])
    elif len(i.split('_')) > 2:
        label_one_val.append(dic_host[i.split('_')[0].lower()])
print(label_one_train)
print(len(label_one_train))

length_train = []
length_val = []
for i in x_train:
    length_train.append(len(i))
# f_length = open('length_revised_add_host_smote_6.txt').readlines()
# for i in f_length:
#     length_train.append(int(i))
for i in x_val:
    length_val.append(len(i))
print(len(length_val))

# f_test = word2vec.LineSentence('data_test_for_align_unaligned_revised_new_add_BA_only_complete.fasta')
# # f_test = word2vec.LineSentence('nonhuman_host_variant.fasta')
# t = 1
# sequence_test = []
# label_test = []
# for i in f_test:
#     if t % 2 == 0:
#         sequence_test.append(i[0])
#     else:
#         label_test.append(i)
#     t += 1
# reviews_test = []
# for i in sequence_test:
#     reviews_test.append(re.findall('.{%d}' % split, i))
# reviews_test = np.asarray(reviews_test)[33776:67552].tolist()
# length_test = []
# for i in reviews_test:
#     length_test.append(len(i))
# print(len(length_test))
# print(len(label_test))

# f_test = word2vec.LineSentence('BA.2_only_complete.fasta')
# # f_test = word2vec.LineSentence('nonhuman_host_variant.fasta')
# t = 1
# sequence_test = []
# label_test = []
# for i in f_test:
#     if t % 2 == 0:
#         sequence_test.append(i[0])
#     else:
#         label_test.append(i)
#     t += 1
# reviews_test = []
# for i in sequence_test:
#     reviews_test.append(re.findall('.{%d}' % split, i))
# length_test = []
# for i in reviews_test:
#     length_test.append(len(i))
# print(len(length_test))


def get_batch(x, y, batchsize, length, training):
    perm = np.arange(x.shape[0])
    y = np.asarray(y)
    length = np.asarray(length)
    if training:
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        length = length[perm]
    numbatch = math.ceil(x.shape[0]/batchsize)
    for i in range(numbatch):
        start = i*batchsize
        end = start+batchsize
        batchx = x[start:end]
        batchy = y[start:end]
        batchlength = length[start:end]
        yield batchx, batchy, batchlength


def attention(h, keep_prob):
    size = hidden_size[-1]
    w = tf.Variable(tf.random_normal([size], stddev=0.1, dtype=tf.float32))
    m = tf.tanh(h, name='m')
    newm = tf.matmul(tf.reshape(m, [-1, size]), tf.reshape(w, [-1, 1]), name='new_m')
    restorem = tf.reshape(newm, [-1, sequence_length], name='restore_m')
    alpha = tf.nn.softmax(restorem, name='alpha')
    r = tf.matmul(tf.transpose(h, [0, 2, 1]), tf.reshape(alpha, [-1, sequence_length, 1]), name='r')
    sequeeze_r = tf.reshape(r, [-1, size])
    repre = tf.tanh(sequeeze_r, name='attn')
    output = tf.nn.dropout(repre, keep_prob=keep_prob, name='h')
    return output, alpha


def build_graph():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, sequence_length, 1024], name='x')
    y = tf.placeholder(tf.int32, [None], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    seq_length = tf.placeholder(tf.int32, name='seq_length')
    embedding = x

    with tf.name_scope('Bi_LSTM'):
        for idx, hiddensize in enumerate(hidden_size):
            with tf.name_scope('Bi-LSTM'+str(idx)):
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddensize),
                                                        output_keep_prob=keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddensize),
                                                        output_keep_prob=keep_prob)
                rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding, dtype=tf.float32,
                                                                scope='bi-lstm'+str(idx), sequence_length=seq_length)
                embedding = tf.concat(rnn_output, 2)
    rnn_output = tf.split(embedding, 2, -1)

    with tf.name_scope('Attention'):
        h = rnn_output[0]+rnn_output[1]
        output = attention(h, keep_prob)[0]
        outputsize = hidden_size[-1]

    with tf.name_scope('output'):
        output_w = tf.get_variable('output_w', shape=[outputsize, 6], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   dtype=tf.float32)
        output_b = tf.Variable(tf.constant(0.1, shape=[6], dtype=tf.float32), name='output_b')
        # l2loss += tf.nn.l2_loss(output_w)
        # l2loss += tf.nn.l2_loss(output_b)
        logits = tf.nn.xw_plus_b(output, output_w, output_b, name='logits')
        prediction = tf.argmax(logits, axis=-1, name='prediction', output_type=tf.int32)

    with tf.name_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(losses)

    correct_predict = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    train_step = opt.minimize(loss, global_step=global_step)
    # train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('H:/Py Files/coronavirus_final_revise/log/', sess.graph)
    return dict(x=x, y=y, keep_prob=keep_prob, loss=loss, train_step=train_step, merged=merged,
                train_writer=train_writer, saver=tf.train.Saver(), prediction=prediction, accuracy=accuracy,
                seq_length=seq_length, alpha=attention(h, keep_prob)[1], logits=logits)


# np.set_printoptions(threshold=np.inf)
# tf.config.set_soft_device_placement(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# start = time.time()
# g = build_graph()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # while f1 < dic_f1[protein]:
#     for i in range(epoch):
#         with tqdm(total=data_train.shape[0]//batchsize) as pbar:
#             for batch in get_batch(data_train, label_one_train, batchsize, length_train, True):
#                 feed_dict = {g['x']: batch[0], g['y']: batch[1], g['keep_prob']: keepprob,
#                              g['seq_length']: batch[2]}
#                 _, loss, accuracy, pred = sess.run([g['train_step'], g['loss'], g['accuracy'], g['prediction']], feed_dict=feed_dict)
#                 pbar.update(1)
#         print('epoch %s accuracy:' % str(i+1), accuracy)
#         feed_dict_val = {g['x']: data_val, g['y']: label_one_val, g['keep_prob']: 1.0, g['seq_length']: length_val}
#         pred_val = sess.run(g['prediction'], feed_dict=feed_dict_val)
#         print(pred_val)
#         print(label_one_val)
#         f1 = metrics.f1_score(label_one_val, pred_val, average='micro')
#         print('F1-score:', f1)
#         i += 1
#         g['saver'].save(sess, 'model_rnn_attention_add_host_6_trying_revised/', i)
# print(time.time()-start)

np.set_printoptions(threshold=np.inf)
ckpt = tf.train.get_checkpoint_state('model_rnn_attention_add_host_6_trying_revised/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    # tf.saved_model.loader.load(sess, ['serve'], 'G:/Py Files/coronavirus/explore/model_rnn_attention_%s_elmo/' % protein)
    g = tf.get_default_graph()
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    x = g.get_operation_by_name('x').outputs[0]
    y = g.get_operation_by_name('y').outputs[0]
    keep_prob = g.get_operation_by_name('keep_prob').outputs[0]
    seq_length = g.get_operation_by_name('seq_length').outputs[0]
    prediction = g.get_operation_by_name('output/prediction').outputs[0]
    output = g.get_operation_by_name('Attention/attn').outputs[0]
    logits = g.get_operation_by_name('output/logits').outputs[0]
    # length_test = length_train+length_val
    # reviews_test = reviews
    data_test = data_val
    reviews_test = x_val
    length_test = length_val
    # data_test = np.load('data_test_BA.2_only_complete.npy')
    # data_test = np.load('data_test_before_omicron_2.npy')
    # data_test = np.load('data_test_nonhuman_host_variant.npy')
    # data_test = np.concatenate([data_train, data_val], axis=0)
    # reviews_test = np.concatenate([x_train, x_val], axis=0)
    # length_test = np.concatenate([length_train, length_val], axis=0)
    out_test = np.zeros([0, hidden_size[-1]])
    # label_pca_test = np.zeros([0])
    pred_test = np.zeros([0])
    prob_test = np.zeros([0, 6])
    with tqdm(total=math.ceil(len(reviews_test) / batchsize)) as pbar:
        print(data_test.shape)
        label_one_test = [10] * data_test.shape[0]
        for i in get_batch(data_test, label_one_test, batchsize, length_test, False):
            feed_dict_test = {x: i[0], y: i[1], keep_prob: 1.0, seq_length: i[2]}
            out_test = np.concatenate([out_test, sess.run(output, feed_dict=feed_dict_test)], axis=0)
            # label_pca_test = np.concatenate([label_pca_test, i[1]], axis=0)
            pred_test = np.concatenate([pred_test, sess.run(prediction, feed_dict=feed_dict_test)], axis=0)
            logit = sess.run(logits, feed_dict=feed_dict_test)
            prob_test = np.concatenate([prob_test, sess.run(tf.nn.softmax(logit))], axis=0)
            pbar.update(1)
    # feed_dict = {x: data_test, y: [6]*data_test.shape[0], keep_prob: 1.0, seq_length: length_test}
    # pred_test = sess.run(prediction, feed_dict=feed_dict)
    # print(prob_test)
    # label_test = np.asarray(label_test)
    # print(label_test[pred_test == 0])
    print(pred_test)
    print(pred_test[pred_test == 0].shape)
    # print(metrics.classification_report(label_one_val, pred_val))
    # np.save('attn_out_train_add_host_revised.npy', out_test)
    # np.save('prediction_train_add_host_revised.npy', pred_test)
    # np.save('probability_train_add_host_revised.npy', prob_test)
    # np.save('attn_out_before_omicron_2_add_host_revised.npy', out_test)
    # np.save('prediction_before_omicron_2_add_host_revised.npy', pred_test)
    # np.save('probability_before_omicron_2_add_host_revised.npy', prob_test)
    # np.save('attn_out_XBB_add_host_revised.npy', out_test)
    # np.save('prediction_XBB_add_host_revised.npy', pred_test)
    # np.save('probability_XBB_add_host_revised.npy', prob_test)
    # np.save('attn_out_nonhuman_variant_add_host_revised.npy', out_test)
    # np.save('prediction_nonhuman_variant_add_host_revised.npy', pred_test)
    # np.save('probability_nonhuman_variant_add_host_revised.npy', prob_test)

    confusion = metrics.confusion_matrix(label_one_val, pred_test)
    print(confusion)
    confusion_rate = np.zeros([6, 6])
    for i in range(confusion.shape[0]):
        summ = copy.deepcopy(np.sum(confusion[i]))
        for j in range(confusion.shape[1]):
            confusion_rate[i][j] = confusion[i][j] / summ
    print(confusion_rate)
    classes = ['PR', 'CH', 'CA', 'AR', 'SU', 'RD']
    plt.figure(dpi=200)
    plt.imshow(confusion_rate, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes, fontsize=14)
    plt.yticks(indices, classes, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    # plt.xlabel('True label', fontsize=14)
    # plt.ylabel('Predicted label', fontsize=14)
    # plt.title('Confusion matrix', fontsize=14)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(second_index - 0.4, first_index+0.1, round(confusion_rate[first_index][second_index], 3),
                     fontsize=14)
    plt.show()
