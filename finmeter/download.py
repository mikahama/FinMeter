from mikatools import *
from uralicNLP import uralicApi
import time
import os

def make_dir(path):
	try:
		os.mkdir(path)
	except:
		pass

def main():
	print("Starting to download... This will take a while")
	print("These models are only needed for semantics, sentiment and metaphors")
	print("If you only need to assess meter and rhyme or hyphenate, you DO NOT need these models")
	print("Sentiment analysis requires tensorflow")
	make_dir(script_path("data"))
	make_dir(script_path("data/metaphor"))
	make_dir(script_path("sentiment/pickle"))
	time.sleep(2)
	files = {"data/metaphor/unigrams_sorted_5k.txt":"https://zenodo.org/record/3473456/files/unigrams_sorted_5k.txt?download=1","data/metaphor/rel_matrix_n_csr.hkl":"https://zenodo.org/record/3473456/files/rel_matrix_n_csr.hkl?download=1","data/fin-word2vec-lemma.bin":"https://zenodo.org/record/3473456/files/fin-word2vec-lemma.bin?download=1", "sentiment/pickle/en.bin": "https://zenodo.org/record/3473456/files/en.bin?download=1","sentiment/pickle/es.bin": "https://zenodo.org/record/3473456/files/es.bin?download=1", "data/fi_concreteness.json":"https://zenodo.org/record/3473456/files/fi_concreteness.txt?download=1"}
	l = len(files.keys())
	i = 0
	for k,v in files.items():
		i = i + 1
		print("Downloading", i, "out of", l )
		print(v, " -->", script_path(k))
		download_file(v, script_path(k), show_progress=True)

	print("Downloading Finnish models for uralicNLP")
	uralicApi.download("fin")

if __name__== "__main__":
  main()