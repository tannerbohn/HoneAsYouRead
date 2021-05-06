
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class LearningModel:

	def __init__(self, sentence_encoder, mode="constant", epsilon=0.2, beta=1, clf_type=None, clf_args=None):

		self.name = "logistic_regression-{}".format(mode)

		self.sentence_encoder=sentence_encoder

		# mode is either constant (epsilon stays same)
		# or "decreasing" (epsilon decreases)
		self.mode = mode

		# the probability of showing (regardless of prediction)
		self.epsilon = epsilon

		self.beta = beta

		self.clf_type = clf_type if clf_type != None else LogisticRegression
		self.clf_args = clf_args if clf_args != None else {}

		return

	def prepare_for_article(self, article):

		sentences = article['sentences']


		
		sent_encs = np.array([self.sentence_encoder.encode(s) for s in sentences])

		self.sentences = sentences
		self.encodings = sent_encs
		self.observed_data = []

		# for the initial model (before we have any feedback), predict all sentences to be shown
		self.model = lambda enc: 1

		self._clf = self.clf_type(*self.clf_args)

		return

	def get_decision(self, sentence_index):

		frac_through = (sentence_index+1) / len(self.sentences)

		#model_show_prob = self.model(self.encodings[sentence_index])

		#show_prob = max(model_show_prob)

		prob_show = self.epsilon if self.mode == "constant" else (1 - frac_through)**self.beta

		#if random.random() < self.epsilon:
		#	return True
		#if frac_through < self.epsilon:
		#	return True
		#if random.random() < (1 - frac_through)**self.epsilon:
		#	return True
		if random.random() < prob_show:
			return True
		else:
			return self.model(self.encodings[sentence_index])


	def incorporate_sample(self, sentence_index, feedback, is_observed):

		model_needs_updating = False

		if is_observed:
			# if is_observed==True, we know we can trust the data
			self.observed_data.append((self.encodings[sentence_index], feedback))
			model_needs_updating = True
		else:
			# if the sample was not observed (ex. it represents a skipped sentence,
			#	we cannot trust it as much)
			pass


		if model_needs_updating:
			labels = list(set([l for _, l in self.observed_data]))
			if len(labels) == 1:
				self.model = lambda enc: labels[0]
			else:
				X, y = list(zip(*self.observed_data))

				self._clf.fit(X, y)
				self.model = lambda enc: self._clf.predict([enc])[0]


		return