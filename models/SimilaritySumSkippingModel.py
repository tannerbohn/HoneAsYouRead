
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

class SimilaritySumSkippingModel:

	def __init__(self, sent_encoder, similarity_threshold=0.5, beta=0, gamma=1):

		self.name = "sim_sim_skipping"

		self.sent_encoder = sent_encoder
		self.similarity_threshold = similarity_threshold
		self.beta = beta
		self.gamma = gamma

		return

	def prepare_for_article(self, article):

		self.nb_sents = len(article['sentences'])

		self.sim_sums = [0 for _ in article['sentences']]

		self.decisions = [True for _ in article['sentences']]

		self.sent_encs = [self.sent_encoder.encode(s) for s in article['sentences']]

		self.similarities = cosine_similarity(self.sent_encs, self.sent_encs)

		return

	def get_decision(self, sentence_index):

		return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed: return

		multiplier = 1 if feedback == 0 else -self.beta

		i = sentence_index + 1
		while i < self.nb_sents:

			#self.sim_sums[i] += multiplier * (max(0, self.similarities[sentence_index][i])**self.gamma)
			sim = self.similarities[sentence_index][i]
			self.sim_sums[i] = max(self.sim_sums[i], sim)

			# if weve already decided not to show it, skip
			if self.sim_sums[i] > self.similarity_threshold: 
				self.decisions[i] = False
			else:
				self.decisions[i] = True

			i += 1


		return

