import tensorflow as tf
from .utils.math import *
from .utils.bdi import *
from .utils.model import *
from . import pickle2 as pickle
from .dan_eval import SentiDAN
from mikatools import script_path

config = tf.ConfigProto()
sess = tf.Session(config=config)
cnn = SentiDAN(sess)
cnn.load(script_path('senti_model.bin'))
infile = script_path('checkpoints/en-es-bimap-1.bin')

MAX_LEN = 64
N = 5
dic = load_model(infile)
W_src = dic['W_source']
W_trg = dic['W_target']
src_lang = dic['source_lang']
trg_lang = dic['target_lang']
model = dic['model']
with open(script_path('pickle/%s.bin' % src_lang), 'rb') as fin:
	src_wv = pickle.load(fin)
with open(script_path('pickle/%s.bin' % trg_lang), 'rb') as fin:
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


def sents2index(X, y, wordvecs, binary):
	X_new = []
	for sent in X:
		sent_new = []
		for word in sent:
			try:
				sent_new.append(wordvecs.word2index(word.lower()))
			except KeyError:
				continue
		X_new.append(sent_new)
	if binary:
		y = (y >= 2).astype(np.int32)
	return X_new, y

def ind2vec(X, y, emb, shuffle, mean):
	if mean:
		size = len(X)
		vec_dim = emb.shape[1]
		X_new = np.zeros((size, vec_dim), dtype=np.float32)
		for i, row in enumerate(X):
			if len(row) > 0:
				X_new[i] = np.mean(emb[row], axis=0)
	else:
		X_new = emb[X]
	if shuffle:
		perm = np.random.permutation(X_new.shape[0])
		X_new, y = X_new[perm], y[perm]
	return X_new, y

def predict(sentences):
	ans = [[s.split()] for s in sentences]
	X = sum(ans, [])

	Y = np.concatenate([np.full(len(t), i)
						for i, t in enumerate(ans)], axis=0)

	X, Y = sents2index(X, Y, trg_wv, binary=False)
	X, Y = ind2vec(X, Y, trg_proj_emb, shuffle=True, mean=True)
	return cnn.predict(X)

#print(predict(["täällä on sika kivaa", "tällä on tylsää ja huonoa"]))
