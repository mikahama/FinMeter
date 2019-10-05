"""
helpers for loading and saving models.

author: fyl
"""
import pickle
import os

def save_model(W_src, W_trg, src_lang, trg_lang, model_type, path, **kwargs):
    dic = {
        "W_source": W_src,
        "W_target": W_trg,
        "source_lang": src_lang,
        "target_lang": trg_lang,
        "model": model_type,
    }
    dic.update(kwargs)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path, 'wb') as fout:
        pickle.dump(dic, fout)


def load_model(path):
    with open(path, 'rb') as fin:
        dic = pickle.load(fin)
    return dic
