#FinMeter

FinMeter is a library for analyzing poetry in Finnish. It handels typical rhyming such as alliteration, assonance and consonance, Japanese meters and Kalevala meter. It can also be used to hyphenate Finnish.

##Hyphenation

Finnish words can be divided into syllables like so

	import finmeter
	print( finmeter.hyphenate("hattu") )
	>> hat-tu
	print( finmeter.syllables("hattu") )
	>> ["hat", "tu"]
	print( finmeter.count_sentence_syllables("kissa juoksi") )
	>> 4

##Rhyming

FinMeter can be used to check whether two words rhyme

	import finmeter
	print( finmeter.assonance("ladata", "ravata") ) #True
	print( finmeter.consonance("kettu", "katti") ) #True
	print( finmeter.full_rhyme("pallolla", "kallolla") ) #True
	print( finmeter.alliteration("voi", "vehnä") ) #True

##Syllabic meters

Meters based on the number of syllables can be assessed by FinMeter

	import finmeter
	print( finmeter.list_possible_meters() )
	>> ['tanka', 'kalevala', 'katauta', 'sedooka', 'bussokusekika', 'haiku', 'chooka']
	print( finmeter.assess_meter("kissa juoksee\nkovaa juoksee", "haiku") )
	>> {'verse_results': [(False, '4/5'), (False, '4/7')], 'poem_length_error': '2/3', 'poem_length_ok': False}

The result is a dictionary cointaining information about the meter for each verse in "verse results" and about the overall length in "poem_length_error". **Note:** For Kalevala you should use *analyze_kalevala* instead.

##Kalevala meter

Kalevala meter functionality takes the poetic foot into account and accepts verses of upto 10 syllables providing that certain poetic rules are met. In addition, the method assess other features important in Kalevala

	import finmeter
	print( finmeter.analyze_kalevala("Vesi vanhin voitehista\nJänö juoksi järveen") )
	>> [{'base_rule': {'message': '', 'result': True}, 'verse': u'Vesi vanhin voitehista', 'normal_meter': True, 'style': {'alliteration': True, 'viskuri': True}}, {'base_rule': {'message': 'Not enough syllables', 'result': False}, 'verse': u'J\xe4n\xf6 juoksi j\xe4rveen', 'style': {'alliteration': True, 'viskuri': True}}]

The method returns a list of analysis results for each verse. If base_rule is True, it means that the verse follows the Kalevala meter, both in syllables and in foot.

##Syllable length

To check if a syllable is short, use the following method

	import finmeter
	print( finmeter.is_short_syllable("tu") )

##Cite

For the time being please idicate that you are using the rhyming functionality of Poem Machine by citing the following publication.

Hämäläinen, M. (2018). Poem Machine - a Co-creative NLG Web Application for Poem Writing. In *The 11th International Conference on Natural Language Generation: Proceedings of the Conference* (pp. 195–196)
