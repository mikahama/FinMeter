#encoding: utf-8
import unidecode

#kotus listing
diphtongs = [u"ai", u"ei", u"oi", u"äi", u"öi", u"ey", u"äy", u"öy", u"au", u"eu", u"ou", u"ui", u"yi", u"iu", u"iy"]
vowels = u"aeiouyäö"

def _remove_diacritics(word):
	r_word = ""
	for c in word:
		if c not in vowels:
			r_word += unidecode.unidecode(c)
		else:
			r_word += c
	return r_word

def hyphenate(word):
	word = _remove_diacritics(word) +" "
	prev_c = ""
	ret = []
	current_syllable = ""
	for i in range(len(word)-1):
		c = word[i]
		if prev_c == "":
			current_syllable += c
		elif prev_c == c and c in vowels:
			current_syllable += c
		elif c not in vowels and word[i+1] in vowels:
			#new syllable
			ret.append(current_syllable)
			current_syllable = c
		elif c in vowels and prev_c in vowels:
			if prev_c +c in [u"ie", u"uo", u"yö"] and len(ret) ==0:
				#diphtong in the beginning
				ret.append(current_syllable + c)
				current_syllable = ""
			elif prev_c +c in diphtongs:
				ret.append(current_syllable + c)
				current_syllable = ""
			else:
				#can't form a diphtong
				ret.append(current_syllable )
				current_syllable = c
		else:
			current_syllable += c
		if len(current_syllable) > 0:
			prev_c = c
		else:
			prev_c = ""
	ret.append(current_syllable)
	ret = _remove_orphan_consonants(ret)
	return "-".join(ret).replace("--", "-")

def _remove_orphan_consonants(syllables):
	output = []
	for i in range(len(syllables)):
		s = syllables[i]
		if len(s) == 1 and s not in vowels:
			if i == 0:
				syllables[1] = syllables[i] +syllables[1]
			else:	
				syllables[i-1] += syllables[i]
			syllables[i] = ""
	for s in syllables:
		if len(s) > 0:
			output.append(s)
	return output

if __name__ == '__main__':
	words = ["kala", "kuitenkin","kurssi","kengät","kalaa", "herttuaa", "köyhien", "puolueita", "paperien", "hygienia","vian", "seassa", "loassa","mullassa", "dia", "karkea","myllyä", "vaa'an","lae", "lammas", "ha", "väinämöinen", "koiravaljakko", "kuorma-auto"]
	for word in words:
		print(word + " " + hyphenate(word))


