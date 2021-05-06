
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class CoverageModel:

	def __init__(self, sent_encoder, n_concepts=4, beta = 0.2, c=0):

		self.name = "coverage"

		self.sent_encoder = sent_encoder

		self.n_concepts = n_concepts

		# adaptation speed
		self.beta = beta

		self.c = c

		return

	def prepare_for_article(self, article):

		self.sentences = article['sentences']
		self.nb_sents = len(article['sentences'])

		self.decisions = [True for _ in article['sentences']]

		self.sent_encs = [self.sent_encoder.encode(s) for s in article['sentences']]

		#self.similarities = cosine_similarity(self.sent_encs, self.sent_encs)

		# compute concept vectors
		cluster = KMeans(n_clusters=self.n_concepts, random_state=0).fit(self.sent_encs)
		centers = cluster.cluster_centers_

		self.concept_vectors = centers

		self.concept_weight_sums = np.ones(self.n_concepts)*self.c #np.zeros(self.n_concepts)
		self.concept_weights = 1/(1+np.exp(-self.concept_weight_sums/self.beta)) # np.ones(self.n_concepts)

		# determine similarity of sentences to concepts
		# sim_matric[sentence_index] = concept similarities
		self.sim_matrix = cosine_similarity(self.sent_encs, self.concept_vectors)

		self.length_frac_estimate = 1
		
		'''
		sent_importances = np.max(self.sim_matrix * self.concept_weights,axis=1)

		indices = list(range(self.nb_sents))
		scored_indices = zip(indices, sent_importances)
		
		scored_indices = sorted(scored_indices, key=lambda k: -k[1])

		sorted_indices, _ = list(zip(*scored_indices))

		score_sorted_sent_indices = sorted_indices

		self.decisions = [False for _ in self.sentences]

		# this part is the same independent of the fixed/dynamic mode
		for i in score_sorted_sent_indices[:int(self.length_frac_estimate*self.nb_sents)]:
			self.decisions[i] = True
		'''
		self.decisions = [True for _ in self.sentences]

		# need this for estimating dynamic length (averaging feedback data)
		self.disliked_sent_indices = []

		self.read_sent_concept_sims = []

		return

	def get_decision(self, sentence_index):

		return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed: return

		self.read_sent_concept_sims.append(self.sim_matrix[sentence_index])

		if feedback == 0:
			self.disliked_sent_indices.append(sentence_index)

			# update concepts weights
			self.concept_weight_sums = self.concept_weight_sums - self.sim_matrix[sentence_index]

		else:

			self.concept_weight_sums = self.concept_weight_sums + self.sim_matrix[sentence_index]

		#self.concept_weights = np.clip(self.concept_weights, 0, 1)
		self.concept_weights = 1/(1+np.exp(-self.concept_weight_sums/self.beta))


		# update decisions...
		# - with new concept weights, look at previously disliked sentences to estimate length preference
		# - and then rank sentences by importance and improved coverage of important sentences (separate weighting
		#	for these aspects)

		if len(self.disliked_sent_indices) == 0: return


		# compute current coverages
		cur_coverage = np.max(self.read_sent_concept_sims, axis=0)
		needed_coverage = (np.ones(self.n_concepts) - cur_coverage) * self.concept_weights
		sent_coverage_scores = np.dot(self.sim_matrix, needed_coverage)
		sent_importances = sent_coverage_scores



		disliked_importances = [sent_importances[i] for i in self.disliked_sent_indices]
		self.length_frac_estimate = 1 - np.average(disliked_importances)


		indices = list(range(self.nb_sents))
		scored_indices = zip(indices, sent_importances)
		
		scored_indices = sorted(scored_indices, key=lambda k: -k[1])

		sorted_indices, _ = list(zip(*scored_indices))

		score_sorted_sent_indices = sorted_indices


		for i in range(self.nb_sents):
			# only modify those decision after* the current reading point
			if i <= sentence_index:
				continue

			if i in score_sorted_sent_indices[:int(self.length_frac_estimate*self.nb_sents)]:
			#if sent_importances[i] >= (1 - self.length_frac_estimate):
				self.decisions[i] = True
			else:
				self.decisions[i] = False



		return

