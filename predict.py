import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
import math
import sys
import argparse
from gensim.models import word2vec
import re

tf.disable_v2_behavior()
tf.disable_eager_execution()


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


parser = argparse.ArgumentParser('Predict SARS-CoV-2 host adaptation with Bi-LSTM model')
parser.add_argument('--model_path', type=str, help='model path of Bi-LSTM model')
parser.add_argument('--num_class', type=int, default=6, help='amount of label classes')
parser.add_argument('--data', type=str, help='file of embedded data to predict')
parser.add_argument('--batchsize', type=int, default=256, help='batch size in predicting')
parser.add_argument('--file', type=str, help='file of sequence length to predict')
parser.add_argument('--n_split', type=int, default=6, help='amount of amino acid in a token')
parser.add_argument('--out_path', type=str, help='path to output the predicting result')
args = parser.parse_args(sys.argv[1:])

ckpt = tf.train.get_checkpoint_state(args.model_path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    g = tf.get_default_graph()
    x = g.get_operation_by_name('x').outputs[0]
    y = g.get_operation_by_name('y').outputs[0]
    keep_prob = g.get_operation_by_name('keep_prob').outputs[0]
    seq_length = g.get_operation_by_name('seq_length').outputs[0]
    logits = g.get_operation_by_name('output/logits').outputs[0]
    prediction = g.get_operation_by_name('output/prediction').outputs[0]
    # output = g.get_operation_by_name('Attention/attn').outputs[0]
    # out_test = np.zeros([0, hidden_size[-1]])
    prob = np.zeros([0, args.num_class])
    label_pca_test = np.zeros([0])
    pred_test = np.zeros([0])
    data = np.load(args.data)
    label = [6]*data.shape[0]

    f_test = word2vec.LineSentence(args.file)
    t = 1
    sequence_test = []
    for i in f_test:
        if t % 2 == 0:
            sequence_test.append(i[0])
        t += 1
    reviews_test = []
    length_test = []
    for i in sequence_test:
        reviews_test.append(re.findall('.{%d}' % args.n_split, i))
        length_test.append(len(re.findall('.{%d}' % args.n_split, i)))
    del f_test

    with tqdm(total=data.shape[0] // args.batchsize) as pbar:
        label_one_test = [args.num_class] * data.shape[0]
        for batch in get_batch(data, label, args.batchsize, length_test, False):
            feed_dict_test = {x: batch[0], y: batch[1], keep_prob: 1.0, seq_length: batch[2]}
            # out_test = np.concatenate([out_test, sess.run(output, feed_dict=feed_dict_test)], axis=0)
            logit = sess.run(logits, feed_dict=feed_dict_test)
            prob = np.concatenate([prob, sess.run(tf.nn.softmax(logit))], axis=0)
            # label_pca_test = np.concatenate([label_pca_test, batch[1]], axis=0)
            pred_test = np.concatenate([pred_test, sess.run(prediction, feed_dict=feed_dict_test)], axis=0)
            pbar.update(1)
    # del args.data
    if args.out_path[-1] != '/':
        outpath = args.out_path+'/'
    else:
        outpath = args.out_path
    np.save(outpath+'prediction_probability.npy', prob)
    print(pred_test[pred_test == 0].shape)
