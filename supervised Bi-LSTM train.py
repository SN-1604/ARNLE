import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
import math
from sklearn import metrics
import sys
import argparse

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


def attention(h, keep_prob):
    size = hidden_size[-1]
    w = tf.Variable(tf.random_normal([size], stddev=0.1, dtype=tf.float32))
    m = tf.tanh(h, name='m')
    newm = tf.matmul(tf.reshape(m, [-1, size]), tf.reshape(w, [-1, 1]), name='new_m')
    restorem = tf.reshape(newm, [-1, max_length], name='restore_m')
    alpha = tf.nn.softmax(restorem, name='alpha')
    r = tf.matmul(tf.transpose(h, [0, 2, 1]), tf.reshape(alpha, [-1, max_length, 1]), name='r')
    sequeeze_r = tf.reshape(r, [-1, size])
    repre = tf.tanh(sequeeze_r, name='attn')
    output = tf.nn.dropout(repre, keep_prob=keep_prob, name='h')
    return output, alpha


def build_graph():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, max_length, 1024], name='x')
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
        output_w = tf.get_variable('output_w', shape=[outputsize, num_class], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   dtype=tf.float32)
        output_b = tf.Variable(tf.constant(0.1, shape=[num_class], dtype=tf.float32), name='output_b')
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
    train_writer = tf.summary.FileWriter(writer_path, sess.graph)
    return dict(x=x, y=y, keep_prob=keep_prob, loss=loss, train_step=train_step, merged=merged,
                train_writer=train_writer, saver=tf.train.Saver(), prediction=prediction, accuracy=accuracy,
                seq_length=seq_length, alpha=attention(h, keep_prob)[1], logits=logits)


parser = argparse.ArgumentParser('Training a Bi-LSTM model')
parser.add_argument('--data_train', type=str, help='data for training the model')
parser.add_argument('--label_train', type=str, help='label of training data')
parser.add_argument('--data_val', type=str, help='data for validate the model')
parser.add_argument('--label_val', type=str, help='label of validation data')
parser.add_argument('--length_train', type=str, help='length file of training data')
parser.add_argument('--length_val', type=str, help='length file of validation data')
parser.add_argument('--epoch', type=int, default=30, help='epoch for training the model')
parser.add_argument('--keepprob', type=float, default=0.8, help='keep probability in dropout')
parser.add_argument('--num_class', type=int, default=6, help='amount of label classes')
parser.add_argument('--hidden_size', type=str, default='256,128', help='hidden layer size of Bi-LSTM model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate when training the model')
parser.add_argument('--max_length', type=int, default=264, help='maximum length of sequence')
parser.add_argument('--writer_path', type=str, help='path to writer training log')
parser.add_argument('--model_path', type=str, help='path to save trained model')
parser.add_argument('--batchsize', type=int, default=256, help='training batchsize')
args = parser.parse_args(sys.argv[1:])

tf.disable_v2_behavior()
tf.disable_eager_execution()
data_train = np.load(args.data_train)
label_train = open(args.label_train).read().splitlines()
dic_host = {'human': 0, 'bat': 1, 'carnivora': 2, 'artiodactyla': 3, 'swine': 4, 'rodentia': 5}
label_train = [dic_host[i.lower().rstrip('\n')] for i in label_train]
data_val = np.load(args.data_val, allow_pickle=True)
# label_val = open(args.label_val).read().splitlines()
# label_val = np.load(args.label_val, allow_pickle=True)
label_val = open(args.label_val).readlines()
label_val = [dic_host[i.lower().rstrip('\n')] for i in label_val]
# label_one_val = [dic_host[i] for i in label_val]
sequence_length = open(args.length_train).read().splitlines()
sequence_length = [int(i.rstrip('\n')) for i in sequence_length]
length_val = open(args.length_val).read().splitlines()
length_val = [int(i.rstrip('\n')) for i in length_val]
# x_val = np.load(args.x_val)
# length_val = []
# for i in x_val:
#     length_val.append(len(i))
num_class = args.num_class
hidden_size = [int(args.hidden_size.split(',')[0]), int(args.hidden_size.split(',')[1])]
lr = args.lr
max_length = args.max_length
writer_path = args.writer_path
print('training data size', data_train.shape)
print('training label size', len(label_train))
print('validation data size', data_val.shape)
print('validation label size', len(label_val))
print(label_train)

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
g = build_graph()
print('model build successful')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(args.epoch):
        with tqdm(total=data_train.shape[0]//args.batchsize) as pbar:
            for batch in get_batch(data_train, label_train, args.batchsize, sequence_length, True):
                feed_dict = {g['x']: batch[0], g['y']: batch[1], g['keep_prob']: args.keepprob,
                             g['seq_length']: batch[2]}
                _, loss, accuracy, pred = sess.run([g['train_step'], g['loss'], g['accuracy'], g['prediction']], feed_dict=feed_dict)
                pbar.update(1)
        # print(pred)
        print('epoch %s accuracy:' % str(i+1), accuracy)
        feed_dict_val = {g['x']: data_val, g['y']: label_val, g['keep_prob']: 1.0, g['seq_length']: length_val}
        pred_val = sess.run(g['prediction'], feed_dict=feed_dict_val)
        f1 = metrics.f1_score(label_val, pred_val, average='micro')
        print('F1-score:', f1)
        g['saver'].save(sess, args.model_path, (i+1))
