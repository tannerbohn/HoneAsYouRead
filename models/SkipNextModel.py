
import numpy as np
import random

class SkipNextModel:

	def __init__(self, n=1):

		self.name = "skip_next_n"#.format(n)

		self.n=n

		return

	def prepare_for_article(self, article):

		self.nb_sents = len(article['sentences'])

		self.decisions = [True for _ in article['sentences']]

		return

	def get_decision(self, sentence_index):

		return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed: return

		if feedback == 0:
			for i in range(sentence_index+1, min(sentence_index+1+self.n,  self.nb_sents)):
				self.decisions[i] = False


		return

