
import numpy as np
import random

class ShowEveryKModel:

	def __init__(self, k=2):

		self.name = "show_every_k"

		self.k = k

		return

	def prepare_for_article(self, article):

		self.nb_sents = len(article['sentences'])

		return

	def get_decision(self, sentence_index):

		return (sentence_index % self.k) == 0

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		return

