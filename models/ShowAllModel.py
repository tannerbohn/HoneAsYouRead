
import numpy as np
import random

class ShowAllModel:

	def __init__(self, p=1):

		self.name = "show_p"

		self.p = p

		return

	def prepare_for_article(self, article):

		return

	def get_decision(self, sentence_index):

		return random.random() < self.p

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		return

