# FinMeter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3474018.svg)](https://doi.org/10.5281/zenodo.3474018) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3473456.svg)](https://doi.org/10.5281/zenodo.3473456)

FinMeter is a library for analyzing poetry in Finnish. It handles typical rhyming such as alliteration, assonance and consonance, Japanese meters and Kalevala meter. It can also be used to hyphenate Finnish and analyse meter. In addition, it can do semantic clustering, metaphor interpretation, concreteness scoring and sentiment analysis

	pip install finmeter

If you use the methods relating to semantics, metaphors and sentiment, you will need to run:

	python3 -m finmeter.download 

Sentiment analysis requires **tensorflow** (tested on 1.9.0 and numpy 1.16.4).

## Hyphenation

Finnish words can be divided into syllables like so

	import finmeter
	finmeter.hyphenate("hattu")
	>> hat-tu
	finmeter.syllables("hattu")
	>> ["hat", "tu"]
	finmeter.count_sentence_syllables("kissa juoksi")
	>> 4

## Rhyming

FinMeter can be used to check whether two words rhyme

	import finmeter
	finmeter.assonance("ladata", "ravata") #True
	finmeter.consonance("kettu", "katti") #True
	finmeter.full_rhyme("pallolla", "kallolla") #True
	finmeter.alliteration("voi", "vehnä") #True

## Syllabic meters

Meters based on the number of syllables can be assessed by FinMeter

	import finmeter
	finmeter.list_possible_meters()
	>> ['tanka', 'kalevala', 'katauta', 'sedooka', 'bussokusekika', 'haiku', 'chooka']
	finmeter.assess_meter(u"kissa juoksee\nkovaa juoksee", "haiku")
	>> {'verse_results': [(False, '4/5'), (False, '4/7')], 'poem_length_error': '2/3', 'poem_length_ok': False}

The result is a dictionary cointaining information about the meter for each verse in "verse results" and about the overall length in "poem_length_error". **Note:** For Kalevala you should use *analyze_kalevala* instead.

## Kalevala meter

Kalevala meter functionality takes the poetic foot into account and accepts verses of upto 10 syllables providing that certain poetic rules are met. In addition, the method assess other features important in Kalevala

	import finmeter
	finmeter.analyze_kalevala(u"Vesi vanhin voitehista\nJänö juoksi järveen")
	>> [{'base_rule': {'message': '', 'result': True}, 'verse': u'Vesi vanhin voitehista', 'normal_meter': True, 'style': {'alliteration': True, 'viskuri': True}}, {'base_rule': {'message': 'Not enough syllables', 'result': False}, 'verse': u'J\xe4n\xf6 juoksi j\xe4rveen', 'style': {'alliteration': True, 'viskuri': True}}]

The method returns a list of analysis results for each verse. If base_rule is True, it means that the verse follows the Kalevala meter, both in syllables and in foot.

## Syllable length

To check if a syllable is short, use the following method

	import finmeter
	finmeter.is_short_syllable("tu") 
	>> True

# Semantics

The library has a variety of different functions realted to semantics

## Concreteness

	from finmeter import semantics

	semantics.concreteness("kissa")
	>> 4.615
	semantics.is_concrete("kissa")
	>> True

The former method outputs True if the concreteness of the word is equal or greater than 3. The latter method outputs a concreteness score from 1 to 5. Both of the methods will return None for out of vocabulary words.

## Semantic clusters

	from finmeter import semantics

	semantics.semantic_clusters(["kissa", "koira", "näätä", "hauki", "vesi", "lemmikki", "puhelin", "tieto|kone", "toimisto"])
	>> [['koira', 'lemmikki', 'kissa', 'näätä'], ['vesi', 'hauki'], ['toimisto', 'tieto|kone', 'puhelin']]
	semantics.similarity_clusters(["koira", "kissa", "hevonen"], ["talo", "koti", "ovi"])
	>> 0.18099508
	semantics.cluster_centroid(["koira", "kissa", "hevonen"])
	>> [-5.84886856e-02 -1.10119150e-03 -3.40119563e-03......]

The library can be used to cluster words together into semantic clusters and to assess the similarity of two word clusters.

# Sentiment

The library provides a somewhat functional sentiment analysis, but I wouldn't hold my breath.

	from finmeter import sentiment
	sentiment.predict("Olipa kakkainen leffa")
	>> -2
	sentiment.predict("Kaikki on tosi kivaa")
	>> 2

The possible values are -2 for strongly negative, -1 for negative, 1 for positive and 2 for strongly positive.

# Metaphors

The library can give interpretations for metaphors. The lower the value, the more likely the interpretation. Example for *mies on susi*

	from finmeter import metaphor
	metaphor.interpret("mies", "susi", maximum=10)
	>> {'A': [('yksinäinen', 0), ('nuori', 3)], 'Adv': [], 'V': [('raadella', 0), ('tappaa', 1), ('ampua', 2), ('liikkua', 2), ('kaataa', 4)], 'N': [('metsästäjä', 1), ('suu', 3), ('vaate', 4)], 'UNK': []}

*maximum* is an optional parameter to limit the number of interpretations. If you do not need POS tagging, you can pass *pos_tags=False*.

# Cite

If you use this library, cite the following publication

Mika Hämäläinen and Khalid Alnajjar (2019). [Let's FACE it. Finnish Poetry Generation with Aesthetics and Framing](https://www.aclweb.org/anthology/W19-8637/). In *the Proceedings of The 12th International Conference on Natural Language Generation*. pages 290-300

# Business solutions


<img src="https://rootroo.com/cropped-logo-01-png/" alt="Rootroo logo" width="128px" height="128px">

Advanced text analysis such as figurative language has a huge potential in many contexts, such as in understanidng the pragmatics of your user data or in generating advertising messages. We are here for you! [Rootroo offers consulting related to advanced text analysis](https://rootroo.com/). We have a strong academic background in the state-of-the-art AI solutions for every NLP need. Just contact us, we won't bite.

