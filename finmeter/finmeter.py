# -*- coding: utf-8 -*-
__author__ = 'mikahama'
import os
import json
from .hyphenate import hyphenate

consonants = ["q","w","r","t","p","s","d","f","g","h","j","k","l","z","x","c","v","b","n","m"]
vowels = ["a","e","i","o","u","y","ä","ö","å"]

def alliteration(word1, word2):
    if len(word1) == 0 or len(word2) == 0:
        return False
    return word1[0].lower() == word2[0].lower()

def assonance(word1, word2):
	return _words_rhyme(word1, word2, consonants)

def consonance(word1, word2):
	return _words_rhyme(word1, word2, vowels)

def _words_rhyme(word1, word2, letters):
	"""
	Checks if two words rhyme when characters in letters variable are omitted
	:param word1: a word "kissa"
	:param word2: another word "kassi"
	:param letters: use consonants for assonance and vowels for consonance e.g ["a","e","i","o","u","y","ä","ö","å"]
	:return: True or False
	"""
	try:
		word1 = word1.lower()
		word2 = word2.lower()

		if word1 == word2:
			#If the words are exactly the same, they don't rhyme
			return False

		for letter in letters:
			word1 = word1.replace(letter, "C")
			word2 = word2.replace(letter, "C")

		#Leading consonants don't matter
		word1 = word1.lstrip("C")
		word2 = word2.lstrip("C")

		if len(word1) < len(word2):
			#make word1 the longer one
			temp = word1
			word1 = word2
			word2 = temp

		if len(word1) > len(word2):
			#The other word can be longer and they'll still rhyme
			difference = len(word1) - len(word2)
			if word1[difference] != "C" and word1[difference-1] != "C":
				#There's a diphthong which can't be cut
				difference = difference-1
			word1 = word1[difference:]

		if word1 == word2:
			return True
		else:
			return False
	except:
		return False

def full_rhyme(word1, word2):
	"""
	Check if words rhyme fully (they are identical from the first vowel onwards) e.g. katti and patti
	:param word1: a word "tatti"
	:param word2: another word "mappi"
	:return: result e.g. False
	"""
	try:
		word1 = word1.lower()
		word2 = word2.lower()

		if word1 == word2:
			#If the words are exactly the same, they don't rhyme
			return False
		while True:
			#strip off leading consonants for word1
			if len(word1) == 0:
				return False
			elif word1[0] in consonants:
				word1 = word1[1:]
			else:
				break
		while True:
			#strip off leading consonants for word2
			if len(word2) == 0:
				return False
			elif word2[0] in consonants:
				word2 = word2[1:]
			else:
				break
		if len(word1) < len(word2):
			#make word1 the longer one
			temp = word1
			word1 = word2
			word2 = temp
		difference = len(word1) - len(word2)
		word1 = word1[difference:]
		if word2 == word1:
			return True
		else:
			return False
	except:
		return False





meter_dict = {}

def list_possible_meters():
    """
    Lists possible meters defined in meters.json
    :return: possible meter names ["haiku", "tanka"...]
    """
    return list(meter_dict.keys())


def syllables(word):
    """
    Returns syllables of a Finnish word
    :param word: a word  "kissa"
    :return: syllables ["kis", "sa"]
    """
    syls = hyphenate(word)
    return syls.split("-")


def count_sentence_syllables(sentence):
    """
    Counts syllables in a sentence
    :param sentence: a sentence "hassu kissa"
    :param lang: en or fi for English and Finnish words respectively
    :return: count e.g. 4
    """
    words = sentence.strip().split(" ")
    c = 0
    for w in words:
        syls = syllables(w)
        c += len(syls)
    return c

def load_meters():
    """
    Loads possible meters from meters.json to the global variable meter_dict
    """
    global meter_dict
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "meters.json")
    f = open(path, "r")
    jsonText = f.read()
    f.close()
    meter_dict = json.loads(jsonText)

load_meters()

def _check_meter(poem, poem_meter, meter_structure={}):
    """
    Checks if a poem follows a meter
    :param poem: a poem "kissa juoksi\ntaloon juoksi" or "a cat ran\nto a house it ran"
    :param poem_meter: name of the meter, use list_possible_meters() to find out supported meters
    :param lang: en or fi for English and Finnish words respectively
    :param meter_structure: a user defined meter dictionary. see meters.json for examples of the format
    :return: results [False, False], (False, "Not long enough")
    """
    if poem_meter in meter_structure.keys():
        meter = meter_structure[poem_meter]
    else:
        meter = meter_dict[poem_meter]
    lines = poem.split("\n")
    lines = list(filter(None, lines))
    repetition_count = 0
    verse_index = 0
    verses = meter["verses"]
    results = []

    end_lines = []

    if "end-verses" in meter:
        #has different rules for final verses
        end_verses = meter["end-verses"]
        for end_verse in end_verses:
            try:
                end_lines.append(lines.pop())
            except:
                return [False, False], (False, "Not long enough")
    end_lines.reverse()

    for line in lines:
        #Check normal verses
        s_count = count_sentence_syllables(line)
        correct_count = verses[verse_index]["syllables"]
        result_string = str(s_count) + "/" + str(correct_count)
        if correct_count == s_count:
            results.append((True, result_string))
        else:
            results.append((False, result_string))
        if verse_index +1 == len(verses):
            verse_index = 0
            repetition_count += 1
        else:
            verse_index += 1

    end_index = 0
    for end_line in end_lines:
        s_count = count_sentence_syllables(end_line)
        correct_count = end_verses[end_index]["syllables"]
        end_index += 1
        result_string = str(s_count) + "/" + str(correct_count)
        if correct_count == s_count:
            results.append((True, result_string))
        else:
            results.append((False, result_string))


    repeat_verses = meter["repeat-verses"]
    if repeat_verses == "no-limit":
        #As many verses as one wants
        return results, (True, "")
    elif repeat_verses == "none":
        #Only a fixed number of verses
        if repetition_count > 0 and verse_index > 0:
            #Too many verses!
            return results, (False, str(len(lines)) + "/" +str(len(verses)) )
        elif len(lines) < len(verses):
            #Too few verses!
            return results, (False, str(len(lines)) + "/" +str(len(verses)))
        else:
            #Correct amount of verses
            return results, (True, "")
    else:
        if repeat_verses.startswith(">"):
            repeat_count = int(repeat_verses[1:])
            if repetition_count > repeat_count:
                return results, (True, "")
            else:
                return results, (False, repeat_verses)

def assess_meter(poem, poem_meter, meter_structure={}):
    a, b = _check_meter(poem, poem_meter, meter_structure=meter_structure)
    r = {}
    r["verse_results"] = a
    r["poem_length_ok"] = b[0]
    r["poem_length_error"] = b[1]
    return r

if __name__ == '__main__':
    print(_check_meter(u"tuo pikkuveli\nnappanahkasen veli\nolen ainoa\noletko fiksuin veli?\nkuinka hassu\nupeat laivat tulee\nolen pörröinen otus", "kalevala"))

