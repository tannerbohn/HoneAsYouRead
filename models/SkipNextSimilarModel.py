
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

class SkipNextSimilarModel:

	def __init__(self, sent_encoder, similarity_threshold=0.8):

		self.name = "skip_next_similar"#_{}".format(similarity_threshold)

		self.sent_encoder = sent_encoder
		self.similarity_threshold = similarity_threshold

		return

	def prepare_for_article(self, article):

		self.nb_sents = len(article['sentences'])

		self.decisions = [True for _ in article['sentences']]

		self.sent_encs = [self.sent_encoder.encode(s) for s in article['sentences']]

		return

	def get_decision(self, sentence_index):

		return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed: return

		if feedback == 0:
			#for i in range(sentence_index+1, min(sentence_index+1+self.n,  self.nb_sents)):
			#	self.decisions[i] = False

			i = sentence_index + 1
			while i < self.nb_sents:
				if cosine_similarity([self.sent_encs[sentence_index]], [self.sent_encs[i]])[0][0] > self.similarity_threshold:
					self.decisions[i] = False
					i += 1
				else:
					break


		return

