import finmeter

print( finmeter.hyphenate("krooninen") )
print(finmeter.is_short_syllable("tu") ) 

"""

from finmeter import semantics

print(semantics.concreteness("kissa"))
print(semantics.is_concrete("kissa"))
print(semantics.semantic_clusters(["kissa", "koira", "näätä", "hauki", "vesi", "lemmikki", "puhelin", "tieto|kone", "toimisto"]))

print(semantics.similarity_clusters(["koira", "kissa", "hevonen"], ["talo", "koti", "ovi"]))
#print(semantics.cluster_centroid(["koira", "kissa", "hevonen"]))


from finmeter import metaphor

print(metaphor.metaphoricity("luovuus", "liekki", ["luovuus", "olla", "liekki", "se", "syttyä", "rinta", "ja", "polttaa"]))
print(metaphor.interpret("aika", "raha")[:10])

"""
from finmeter import sentiment

print(sentiment.predict(["täällä on sika kivaa"]))
print(sentiment.predict(["tällä on tylsää ja huonoa"]))