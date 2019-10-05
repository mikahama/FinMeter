from . import lwvlib
from mikatools import *
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
import numpy as np
import itertools
from scipy import spatial
import statistics


w2v_model = lwvlib.load(script_path("data/fin-word2vec-lemma.bin"), 10000, 500000)

concrete = json_load(script_path("data/fi_concreteness.json"))

_open_class = set(["ADJ", "ADV", "NOUN", "INTJ", "PROPN", "VERB"])


def is_concrete(word, threshold=3):
    conc = concreteness(word)
    if conc is None:
        return None
    elif conc >= threshold:
        return True
    else:
        return False


def concreteness(word):
    if word not in concrete:
        return None
    return statistics.mean(concrete[word])


def is_open_class(ud_pos):
    return ud_pos in _open_class


def _filter_w2v(words):
    return [w for w in words if w in w2v_model]


def _get_matrix(words):
    m = []
    for word in words:
        r = []
        for w in words:
            r.append(w2v_model.similarity(word, w))
        m.append(r)
    return m


def semantic_clusters(lemmas, unique=True):
    words = lemmas
    if unique:
        words = list(set(lemmas))
    words = _filter_w2v(words)
    m = np.array(_get_matrix(words))
    agg = AffinityPropagation(affinity="precomputed")
    u = agg.fit_predict(m)
    return _group_words(words, agg.labels_)


def _group_words(words, labels):
    groups = []
    for x in range(len(set(labels))):
        groups.append([])
    for i, label in enumerate(labels):
        groups[label].append(words[i])
    return groups


def similarity_clusters(c1,c2):
    return np.dot(cluster_centroid(c1), cluster_centroid(c2))

def cluster_centroid(cluster):
    vectors = []
    for word in cluster:
        vectors.append(w2v_model.w_to_normv(word))
    return np.mean(vectors, axis=0)
