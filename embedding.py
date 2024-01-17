from elmoformanylangs import Embedder
from gensim.models import word2vec
import sys
import numpy as np
import math
import copy
from tqdm import tqdm
import argparse
import re

parser = argparse.ArgumentParser('Embedding amino acid sequences')
parser.add_argument('--file', type=str, help='input amino acid sequence file')
parser.add_argument('--input_type', type=str, choices=['npy', 'fasta'], help='input file type')
parser.add_argument('--model_path', type=str, help='model path')
parser.add_argument('--output', type=str, help='output file of embedded data')
parser.add_argument('--batchsize', type=int, default=256, help='batchsize to calculate embedding')
parser.add_argument('--max_length', type=int, help='maximum length of the specific protein sequnece')
parser.add_argument('--split', type=int, default=6, help='amino acid amount of a token')
args = parser.parse_args(sys.argv[1:])

elmo = Embedder(args.model_path)

if args.input_type == 'npy':
    x = np.load(args.file)
    print(x.shape[0])
    data = []
    for b in range(math.ceil(x.shape[0]/args.batchsize)):
        for i in elmo.sents2elmo(x[b*args.batchsize:(b+1)*args.batchsize]):
            origin = copy.deepcopy(i)
            origin = origin.astype(np.float32)
            while origin.shape[0] < args.max_length:
                origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
            if origin.shape[0] > args.max_length:
                origin = origin[:args.max_length]
            data.append(origin)
    data = np.asarray(data)
    print(data.shape)
    np.save(args.output, data)
elif args.input_type == 'fasta':
    f_test = word2vec.LineSentence(args.file)
    t = 1
    sequence_test = []
    for i in f_test:
        if t % 2 == 0:
            sequence_test.append(i[0])
        t += 1
    reviews = []
    for i in sequence_test:
        reviews.append(re.findall('.{%d}' % args.split, i))
    print(len(reviews))

    data = []
    with tqdm(total=math.ceil(len(sequence_test)/args.batchsize)) as pbar:
        for b in range(math.ceil(len(sequence_test)/args.batchsize)):
            for i in elmo.sents2elmo(reviews[b*args.batchsize:(b+1)*args.batchsize]):
                origin = copy.deepcopy(i)
                origin = origin.astype(np.float32)
                while origin.shape[0] < args.max_length:
                    origin = np.concatenate([origin, np.zeros([1, 1024], dtype='float32')], axis=0)
                if origin.shape[0] > args.max_length:
                    origin = origin[:args.max_length]
                data.append(origin)
            pbar.update(1)
    data = np.asarray(data)
    print(data.shape)
    np.save(args.output, data)
