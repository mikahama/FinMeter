from mikatools import *
from .meta4meaning_fi import Meta4meaningFi
from uralicNLP import uralicApi


rows_path = script_path('data/metaphor/unigrams_sorted_5k.txt')
matrix_path = script_path('data/metaphor/rel_matrix_n_csr.hkl')
m4m = Meta4meaningFi(rows_path=rows_path, matrix_path=matrix_path)


def metaphoricity(tenor, vehicle, expression, k=0):
    return m4m.metaphoricity(tenor, vehicle, expression, k=k)

def interpret(tenor, vehicle, pos_tags=True, maximum=None):
    res = m4m.interpret(tenor, vehicle)
    if maximum:
    	res = res[:maximum]
    if pos_tags:
    	return _pos_tag(res)
    else:
    	return res

def _merge_compound_analysis(tags):
	ts = tags.split("#")
	tag = ts[0].split("+")
	for t in range(1,len(ts)):
		tag[0] += ts[t].split("+")[0]
	return tag

def _pos_tag(words):
	pos_tags = {"A":[], "Adv":[], "V":[], "N":[], "UNK":[]}
	accepted_tags = set(pos_tags.keys())
	for word in words:
		analysis = uralicApi.analyze(word[0], "fin", force_local=True)
		tag = "UNK"
		for analys in analysis:
			analys = _merge_compound_analysis(analys[0])
			if word[0] == analys[0] and analys[1] in accepted_tags:
				tag = analys[1]
				break
		pos_tags[tag].append(word)
	return pos_tags
