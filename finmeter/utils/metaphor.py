import io, math
from pprint import pprint
from collections import defaultdict
from tqdm import tqdm
import hickle as hkl
import numpy as np


def read_ngrams(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.replace('|', '')
            l = l.rstrip('\n')  # remove newline
            ngram, freq = l.split('\t')
            grams = ngram.split(' ')
            yield grams, int(freq)


def load_termidx(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        rows = map(lambda r: r.rstrip('\n').split('\t')[0], rows)
        return dict([(w, i) for i, w in enumerate(rows)])


def load_unigram(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        rows = map(lambda r: r.rstrip('\n').split('\t'), rows)
        rows = map(lambda r: (r[0], int(r[1])), rows)
        return dict(rows)


def simple_ll(k, O, f1, f2, N):
    # simple-ll = 2*(O*log(O/E)-(O-E))
    # k: window size
    # O (observer): co-occurence
    # f1 and f2: marginal frequencies
    # N: sample size -> corpus size
    # E (expected): k*f1*f2/N
    E = k * f1 * (f2 / N)
    ll = 0.0
    # convert into one-sided, and clap negative values to 0
    if O > E:
        ll = 2 * (O * math.log(O / E) - (O - E))
    return round(math.log(ll), 8) if ll > 0 else ll
