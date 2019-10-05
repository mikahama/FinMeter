from mikatools import *
from .meta4meaning_fi import Meta4meaningFi


rows_path = script_path('data/metaphor/unigrams_sorted_5k.txt')
matrix_path = script_path('data/metaphor/rel_matrix_n_csr.hkl')
m4m = Meta4meaningFi(rows_path=rows_path, matrix_path=matrix_path)


def metaphoricity(tenor, vehicle, expression, k=0):
    return m4m.metaphoricity(tenor, vehicle, expression, k=k)

def interpret(tenor, vehicle):
    return m4m.interpret(tenor, vehicle)