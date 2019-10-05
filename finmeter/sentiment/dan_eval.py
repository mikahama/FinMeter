import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from .utils.dataset import *
from .utils.math import *
from .utils.bdi import *
from .utils.model import *
import glob
import os
import logging
from . import pickle2 as pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

MAX_LEN = 64
TMP_FILE = 'tmp/dan{:d}.ckpt'.format(np.random.randint(0, 1e10))
N = 5
HEADER = 'infile,src_lang,trg_lang,model,is_binary,' + 'f1_macro,' * N + 'f1_macro,average_dev\n'
TEMPLATE = '{},{},{},{},{},' + '{:.4f},' * N + '{:.4f},{:.4f}\n'


class SentiDAN(object):
    """
    CNN for sentiment classification.
    """

    def __init__(self, sess,
                 vec_dim=300,
                 nclasses=4,
                 learning_rate=0.001,
                 batch_size=50,
                 num_epoch=200,
                 dropout=0.3,
                 C=1e-4):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.dropout = dropout
        self.C = C
        self._build_graph()
        self.initialize()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.batch_weights = tf.placeholder(tf.float32, shape=(None,))
        self.labels = tf.placeholder(tf.int32, shape=(None,))

        W1 = tf.get_variable('W1', (self.vec_dim, self.vec_dim), tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1', (self.vec_dim), tf.float32, initializer=tf.zeros_initializer())

        L1 = self.inputs @ W1 + b1

        W2 = tf.get_variable('W2', (self.vec_dim, self.nclasses), tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', (self.nclasses), tf.float32, initializer=tf.zeros_initializer())
        logits = L1 @ W2 + b2

        self.pred = tf.argmax(logits, axis=1)
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, self.nclasses), logits, weights=self.batch_weights)
        self.loss = self.loss + (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) * self.C
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self, path):
        self.saver.restore(self.sess, path)

    def fit(self, train_x, train_y, train_l, dev_x=None, dev_y=None, weights=None):
        max_f1 = 0
        self.saver.save(self.sess, TMP_FILE)
        nsample = len(train_x)
        if weights is None:
            weights = np.ones(train_x.shape[0])

        lmask = np.zeros(train_x.shape[:2])
        for i, l in enumerate(train_l):
            lmask[i][:l] = 1

        for epoch in range(self.num_epoch):
            loss = 0.
            pred = np.zeros(nsample)
            for index, offset in enumerate(range(0, nsample, self.batch_size)):
                xs = train_x[offset:offset + self.batch_size]
                ys = train_y[offset:offset + self.batch_size]
                ws = weights[offset:offset + self.batch_size]
                mask = np.random.rand(*xs.shape[:2]) > self.dropout
                xs = xs * mask[:, :, np.newaxis]
                ls = (lmask[offset:offset + self.batch_size] * mask).sum(1)
                xs = xs.sum(1) / (ls[:, np.newaxis] + 1e-8)
                _, loss_, pred_ = self.sess.run([self.optimizer, self.loss, self.pred],
                                                {self.inputs: xs,
                                                 self.labels: ys,
                                                 self.batch_weights: ws})
                loss += loss_ * len(xs)
                pred[offset:offset + self.batch_size] = pred_
            loss /= nsample
            fscore = f1_score(train_y, pred, average='macro')

            if dev_x is not None and dev_y is not None:
                dev_f1 = self.score(dev_x, dev_y)
                print('epoch: {:d}  f1: {:.4f}  loss: {:.6f}  dev_f1: {:.4f}\r'.format(epoch, fscore, loss, dev_f1), end='', flush=True)
                if dev_f1 > max_f1:
                    max_f1 = dev_f1
                    self.saver.save(self.sess, TMP_FILE)
            else:
                print('epoch: {:d}  f1: {:.4f}  loss: {:.6f}\r'.format(epoch, fscore, loss), end='', flush=True)
        print()
        self.best_score_ = max_f1
        if dev_x is not None and dev_y is not None:
            self.saver.restore(self.sess, TMP_FILE)

    def predict(self, test_x):
        pred = self.sess.run(self.pred, {self.inputs: test_x, self.keep_prob: 1.})
        return pred

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()

    def save(self, savepath):
        self.saver.save(self.sess, savepath)


def main(args):
    print(str(args))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as fout:
            fout.write(HEADER)

    if args.setting == 'both':
        settings = [True, False]
    elif args.setting == 'binary':
        settings = [True, ]
    elif args.setting == '4-class':
        settings = [False, ]

    for infile in args.W:
        dic = load_model(infile)
        W_src = dic['W_source']
        W_trg = dic['W_target']
        src_lang = dic['source_lang']
        trg_lang = dic['target_lang']
        model = dic['model']
        with open('pickle/%s.bin' % src_lang, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open('pickle/%s.bin' % trg_lang, 'rb') as fin:
            trg_wv = pickle.load(fin)
        src_pad_id = src_wv.add_word('<pad>', np.zeros(src_wv.vec_dim, dtype=np.float32))
        trg_pad_id = trg_wv.add_word('<pad>', np.zeros(trg_wv.vec_dim, dtype=np.float32))
        src_proj_emb = np.empty(src_wv.embedding.shape, dtype=np.float32)
        trg_proj_emb = np.empty(trg_wv.embedding.shape, dtype=np.float32)
        if model == 'ubise':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            length_normalize(src_proj_emb, inplace=True)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
            length_normalize(trg_proj_emb, inplace=True)
        elif model == 'ubi':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        elif model == 'blse':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        else:
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)

        for is_binary in settings:
            src_ds = SentimentDataset('datasets/%s/opener_sents/' % src_lang).to_index(src_wv, binary=is_binary).pad(src_pad_id, MAX_LEN)
            trg_ds = SentimentDataset('datasets/%s/opener_sents/' % trg_lang).to_index(trg_wv, binary=is_binary).to_vecs(trg_proj_emb, shuffle=True)
            vec_dim = src_proj_emb.shape[1]

            train_x = src_proj_emb[src_ds.train[0]]
            train_y = src_ds.train[1]
            train_l = src_ds.train[2]
            perm = np.random.permutation(train_x.shape[0])
            train_x, train_y, train_l = train_x[perm], train_y[perm], train_l[perm]
            dev_x = np.concatenate((trg_ds.train[0], trg_ds.dev[0]), axis=0)
            dev_y = np.concatenate((trg_ds.train[1], trg_ds.dev[1]), axis=0)
            test_x = trg_ds.test[0]
            test_y = trg_ds.test[1]

            class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
            weights = np.zeros(train_x.shape[0], dtype=np.float32)
            for t, w in enumerate(class_weight):
                weights[train_y == t] = w
            # if is_binary:
            #     weights[:] = 1

            test_scores = []
            dev_scores = []
            for i in range(N):
                best_dev = 0
                test_f1 = None
                test_pred = None
                for C in args.C:
                    print('C = {}'.format(C))
                    tf.reset_default_graph()
                    with tf.Session(config=config) as sess:
                        cnn = SentiDAN(sess, vec_dim, (2 if is_binary else 4),
                                       args.learning_rate, args.batch_size, args.epochs, args.dropout, C)
                        cnn.initialize()
                        cnn.fit(train_x, train_y, train_l, dev_x, dev_y, weights)
                        pred = cnn.predict(test_x)
                        score = f1_score(test_y, pred, average='macro')
                        if cnn.best_score_ > best_dev:
                            best_dev = cnn.best_score_
                            test_f1 = score
                            test_pred = pred
                        print('test f1 = {:.4f}'.format(score))
                        cnn.save("./senti_model.bin")
                        quit()

                test_scores.append(test_f1)
                dev_scores.append(best_dev)
            avg_test_f1 = sum(test_scores) / N
            avg_dev_f1 = sum(dev_scores) / N


            print('------------------------------------------------------')
            print('Is binary: {}'.format(is_binary))
            print('Result for {}:'.format(infile))
            print('Test f1 scores: {}'.format(test_scores))
            print('Average test f1 macro: {:.4f}'.format(avg_test_f1))
            print('Average dev score: {:.4f}'.format(best_dev))
            print('Confusion matrix:')
            print(confusion_matrix(test_y, test_pred))
            print('------------------------------------------------------')

            if args.output is not None:
                with open(args.output, 'a', encoding='utf-8') as fout:
                    fout.write(TEMPLATE.format(infile, src_lang, trg_lang, model,
                                               is_binary, *test_scores, avg_test_f1, avg_dev_f1))
    # for f in glob.glob(TMP_FILE + '*'):
    #     os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        help='checkpoint files')
    parser.add_argument('--setting',
                        choices=['binary', '4-class', 'both'],
                        default='4-class',
                        help='classification setting')
    parser.add_argument('-lr', '--learning_rate',
                        help='learning rate (default: 0.001)',
                        type=float,
                        default=0.001)
    parser.add_argument('-e', '--epochs',
                        help='training epochs (default: 100)',
                        default=100,
                        type=int)
    parser.add_argument('-bs', '--batch_size',
                        help='training batch size (default: 50)',
                        default=50,
                        type=int)
    parser.add_argument('--dropout',
                        help='dropout rate (default: 0.3)',
                        default=0.3,
                        type=float)
    parser.add_argument('-C', '--C',
                        nargs='+',
                        type=float,
                        default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        help='regularization parameter')
    parser.add_argument('-o', '--output',
                        help='output file')
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)

    parser.set_defaults(clip=True)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main(args)
