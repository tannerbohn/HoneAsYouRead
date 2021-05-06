
import numpy as np
import random

from .SummaryRanker import SummaryRanker

class GenericSummarizationModel:

	def __init__(self, mode='fixed', length_frac=0.5, epsilon=0.2, summarizer="TextRankSummarizer"):

		self.mode = mode 

		if mode == "fixed":
			self.name = "generic_summ_{}".format(mode)
			self.length_frac = length_frac
			#self.length_frac_estimate = self.length_frac
		elif mode == "dynamic":
			self.name = "generic_summ_{}".format(mode)
			self.epsilon = epsilon
			#self.length_frac_estimate = 1.0


		self.summarizer_name = summarizer
		self.summary_ranker = SummaryRanker(summarizer)

		#print("length frac est:", self.length_frac_estimate)

		return

	def prepare_for_article(self, article):

		# reset initial length estimates
		if self.mode == "fixed":
			self.length_frac_estimate = self.length_frac
		elif self.mode == "dynamic":
			self.length_frac_estimate = 1

		self.nb_sents = len(article['sentences'])
		self.sentences = article['sentences']

		self.sent_summ_scores = self.summary_ranker.predict_article(article)


		# compute sentence indices sorted by score (from high to low)
		indices = list(range(self.nb_sents))
		scored_indices = zip(indices, self.sent_summ_scores)
		
		scored_indices = sorted(scored_indices, key=lambda k: -k[1])

		sorted_indices, _ = list(zip(*scored_indices))

		self.score_sorted_sent_indices = sorted_indices

		self.decisions = [False for _ in self.sentences]

		# this part is the same independent of the fixed/dynamic mode
		for i in self.score_sorted_sent_indices[:int(self.length_frac_estimate*self.nb_sents)]:
			self.decisions[i] = True


		# need this for estimating dynamic length (averaging feedback data)
		self.disliked_sent_importances = []

		return

	def get_decision(self, sentence_index):

		if self.mode == "fixed":
			return self.decisions[sentence_index]
		elif self.mode == "dynamic":

			if random.random() < self.epsilon:
				return True
			else:
				return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed: return

		if self.mode == "fixed": return

		# otherwise with dynamic length estimation...
		if feedback == 0:
			#print("got feedback", sentence_index, feedback)
			importance = self.sent_summ_scores[sentence_index]

			self.disliked_sent_importances.append(importance)

			#print(self.disliked_sent_importances)

			# update the length estimation

			self.length_frac_estimate = 1 - np.average(self.disliked_sent_importances)

			for i in range(self.nb_sents):
				# only modify those decision after* the current reading point
				if i <= sentence_index:
					continue

				if i in self.score_sorted_sent_indices[:int(self.length_frac_estimate*self.nb_sents)]:
					self.decisions[i] = True
				else:
					self.decisions[i] = False


		return

