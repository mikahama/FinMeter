# -*- coding: utf-8 -*-
__author__ = 'mikahama'
from .finmeter import syllables as get_syllables
import codecs


vowels = u"aeiouyäöå"
ok_chars = u"qwertyuiopåäölkjhgfdsazxcvbnm'- "

def analyse_style(words):
    """
    Outputs an analysis of the style of the verse (viskuri law and alliteration)
    :param words: ["vesi", "vanhin", "voitehista"]
    :return: a dictionary of the style {"viskuri":True, "alliteration": True}
    """
    return_dict = {}
    first_letters = []
    lengths = []
    for w in words:
        if w[0] == u"å":
           first_letters.append(u"o")
        else:
            first_letters.append(w[0])
        lengths.append(len(w))
    if len(first_letters) != len(set(first_letters)):
        #Alliteration
        return_dict["alliteration"] = True
    else:
        return_dict["alliteration"] = False
    last_len = lengths[len(lengths)-1]
    last_at_end = True
    for i in range(len(lengths)-1):
        w = lengths[i]
        if w > last_len:
            #longest word is not at the end
            last_at_end = False
            break
    return_dict["viskuri"] = last_at_end
    return return_dict





def is_short_syllable(syllable):
    """
    Checks if a syllable is short
    :param syllable: a syllable "au"
    :return: True or False
    """
    syllable.replace(u"*",u"")
    syllable.replace(u"!",u"")
    syllable.replace(u"_",u"")
    syllable = syllable[::-1].lower()
    if len(syllable) == 0:
        return False
    if syllable[0] in vowels:
        if len(syllable) > 1:
            if syllable[1] in vowels:
                #ends with two vowels
                return False
            else:
                #ends with a vowel and a consonant
                return True
        else:
            #only one vowel
            return True
    else:
        #ends with a consonant
        return False

def is_normal_meter(feet):
    """
    Checks if the meter is normal (first syllables are at the beginning of a foot)
    :param feet: a list of feet ["vesi", "vanhin" "voite", "hista"]
    :return: True or False
    """
    for foot in feet:
        for i in range(len(foot)):
            syllable = foot[i]
            if "*" in syllable and i != len(foot)-1:
                #First syllable in another position than at the beginning of a foot
                return False
    return True

def ends_in_long_vowel(syllable):
    """
    Checks if a syllable ends in a long vowel
    :param syllable: a syllable "koo"
    :return: Result e.g. True
    """
    l = len(syllable)
    if l < 2:
        return False
    if syllable[l-1] in vowels and syllable[l-1] == syllable[l-2]:
        return True
    else:
        return False

def base_rule(feet):
    """
    Checks is the feet follow Kalevala's base rules
    :param feet: a list of feet ["vesi", "vanhin" "voite", "hista"]
    :return: result of the analysis {"message": "", "result": True}
    """
    return_data = {}
    feet = feet[:3]
    if u"!" in feet[2][1]:
        return_data["message"] = "4 syllabic word in the second foot"
        return_data["result"] = False
    elif ends_in_long_vowel(feet[0][0]):
        return_data["message"] = "Verse ends in a long vowel"
        return_data["result"] = False
    elif u"_" in feet[0][0]:
        return_data["message"] = "Verse ends in a monosyllabic word"
        return_data["result"] = False
    else:
        return_data["message"] = ""
        return_data["result"] = True
        for i in range(len(feet)):
            foot = feet[i]
            if u"*" in foot[1] and is_short_syllable(foot[1]):
                #should be long!
                return_data["message"] = "Foot #" + str(i+1) + " starts with a word in which the first syllable is short"
                return_data["result"] = False
            elif u"*" in foot[0] and not is_short_syllable(foot[0]):
                return_data["message"] = "Foot #" + str(i+1) + " ends with a word in which the first syllable is long"
                return_data["result"] = False

    return return_data

def analyse(verse_o):
    """
    Analyses a verse
    :param verse_o: a verse "Vesi vanhin voitehistani"
    :return: analysis as a dictionary {"verse" :"Vesi vanhin voitehistani" "message":"Too many syllables": "style":{"viskuri":True, "alliteration": True..}...}
    """
    return_dict = {"verse": verse_o}
    verse = ""
    verse_o = verse_o.lower()
    for c in verse_o:
        if c in ok_chars:
            verse += c
    words = verse.strip().split(" ")
    return_dict["style"] = analyse_style(words)
    sentence = []
    for word in words:
        syls = get_syllables(word)
        if len(syls) == 4:
            #let's mark words that are  4 syllables long
            syls[0] = u"!" +syls[0]
        if len(syls) > 1:
            #Monosyllabic words don't follow the base rule, hence if > 1, for other words, let's mark the first syllable with a star *
            syls[0] = u"*" +syls[0]
        elif len(syls) == 1:
            #Monosyllabic words are marked in a different way
            syls[0] = u"_" +syls[0]
        sentence.extend(syls)

    sentence = sentence[::-1]
    if len(sentence) < 8:
        return_dict["base_rule"] = {"message": "Not enough syllables", "result": False}
    elif len(sentence) > 10:
        return_dict["base_rule"] = {"message": "Too many syllables", "result": False}
    else:
        feet = [sentence[0:2], sentence[2:4], sentence[4:6], sentence[6:]]
        return_dict["normal_meter"] = is_normal_meter(feet)
        return_dict["base_rule"] = base_rule(feet)
    return return_dict

def analyse_verses(text):
    """
    Analyses verses in a text
    :param text: A poem's text e.g. "Vesi vanhin voitehista\nLaulo kerran lauloi toisen\nlauloi vielä kolmannenkin"
    :return: a list of analysis results [{"verse" :"Vesi vanhin voitehista"...}...]
    """
    lines = text.split("\n")
    results = []
    for line in lines:
        if line == "":
            continue
        else:
            results.append(analyse(line))
    return results

if __name__ == "__main__":
    print(analyse(u"Vaka vanha Väinämöinen"))


