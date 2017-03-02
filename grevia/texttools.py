#!/usr/bin/python
# -*- coding: utf-8 -*-


""" Tools for extracting information in texts. """

from collections import Counter
import pandas as pd
from functools import reduce

def proper_nouns(list_of_texts,error_rate=10,noun_occur_threshold=1):
	""" Return a dic of nouns found in the given list of texts.
		A word is a valid noun if: 
		the number of occurences of the capitalized version of the word is high the the uncapitalized version
		the number of uncapitalized occurence is lower 'error_rate'
		the number of capitalized versions is higher than 'noun_occur_threshold'
		The values in the returned dic is a list with:
		 first number is the occurence of the capitalized version
		 second number is the occurence of the uncapitalized version
	"""

	capital_list = []
	non_capital_list = []
	# find the word in capitals
	print("Collecting words in capitals...")
	for text in list_of_texts:
		if isinstance(text, str):
			text = text.split(' ')
		capital_list= capital_list + [word for word in text if word.istitle()]
	capital_dic = Counter(capital_list)
	capital_set_lower = set([ word.lower() for word in capital_dic.keys()])
	# find the non capital version of the capitalized words in the texts
	print('Collecting words not in capitals...')
	for text in list_of_texts:
		if isinstance(text, str):
			text = text.split(' ')
		common_words = set(text) & capital_set_lower
		if common_words:
			non_capital_list = non_capital_list + [word for word in text if word in common_words]
	non_capital_dic = Counter([ word.title() for word in non_capital_list])
	# Keep only the nouns that have more occurence in Capitals
	print('Filtering words...')
	filtered_capital_dic = {}
	for word in capital_dic.keys():
		#print(capital_dic[word],non_capital_dic[word])
		if (non_capital_dic[word]<capital_dic[word] and 
			non_capital_dic[word]<error_rate and capital_dic[word]>noun_occur_threshold):
			filtered_capital_dic[word] = [capital_dic[word],non_capital_dic[word]]

	return filtered_capital_dic

def capital_list(list_of_words):
	return [word for word in list_of_words if word.istitle()]

def capital_list_corpus(iterable_of_texts):
	list_of_capitals = reduce(sum,(map(capital_list,iterable_of_texts)))
	return Counter(list_of_capitals)

#def reduce_capital_list(list_of_words1,list_of_words2):
#	list_of_words = list_of_words1 + list_of_words2
#	return Counter(list_of_words)