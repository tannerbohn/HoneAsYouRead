
import numpy as np
import random

class ShowFirstModel:

	def __init__(self, frac=0.5):

		self.name = "show_first"

		self.frac = frac

		return

	def prepare_for_article(self, article):

		self.nb_sents = len(article['sentences'])

		return

	def get_decision(self, sentence_index):

		return sentence_index <= self.frac * self.nb_sents
		#return sentence_index < self.n

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		return

