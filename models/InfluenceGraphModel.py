
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA

from .SummaryRanker import SummaryRanker

class InfluenceGraphModel:

	def __init__(self, sentence_encoder, initial_value=0.1, skipped_value=-0.1, accepted_value=0.1, disliked_value=-1):

		self.name = "influence_graph"

		self.sentence_encoder = sentence_encoder

		self._initial_value = initial_value # or >= 0
		self._skipped_value = skipped_value # or < 0
		self._accepted_value = accepted_value # or 0 or 1?
		self._disliked_value = disliked_value

		return

	def prepare_for_article(self, article):


		self.influence_graph = InfluenceGraph(article, self.sentence_encoder, self._initial_value, self._skipped_value, self._accepted_value, self._disliked_value)

		return

	def get_decision(self, sentence_index):

		sentence_decision = self.influence_graph.get_sentence_decision(sentence_index)

		return sentence_decision

	def incorporate_sample(self, sentence_index, label, is_observed):

		self.influence_graph.add_feedback(sentence_index, is_observed=is_observed, label=label)

		return





class InfluenceGraph:

	def __init__(self, article, sentence_encoder, initial_value=0.1, skipped_value=-0.1, accepted_value=0.1, disliked_value=-1):

		# initial value should be > 0 so that it is not immediately skipped because of
		# 	a disliked sentence only slightly similar to the one in question
		self._initial_value = initial_value # or >= 0
		# - skipped value should be 0 because we have no real info about it
		# - skipped value should be < 0 because we have some evidence against it
		self._skipped_value = skipped_value # or < 0
		self._accepted_value = accepted_value # or 0 or 1?
		self._disliked_value = disliked_value


		self.article = article
		self.nb_sentences = len(self.article['sentences'])

		# define default values to be slightly positive
		sentences = article['sentences']
		self.node_values = [self._initial_value for _ in sentences]


		self.edge_weights = get_influence_graph(article, sentence_encoder)

		return

	def add_feedback(self, sentence_index, is_observed, label):

		# status: "skipped", "disliked", "accepted"

		# TODO: try different value for skipped sentences/nodes

		# rating: +1 if shown and accepted, -1 if rejected, 0 if not shown



		if not is_observed:
			self.node_values[sentence_index] = self._skipped_value
		elif is_observed and label == 1:
			self.node_values[sentence_index] = self._accepted_value
		elif is_observed and label == 0:
			self.node_values[sentence_index] = self._disliked_value

		# TODO: we *should* now figure out the repercussions of this effect
		#	- i.e. what sentences does the value change remove
		
		for i in range(sentence_index+1, self.nb_sentences):
			current_value = self.node_values[i]

			influence_sum = 0

			for j in range(i):
				v = self.node_values[j]
				w = self.edge_weights[j][i]

				influence_sum += v*w

			score = (influence_sum + current_value)

			if score < 0:
				self.node_values[i] = self._skipped_value
			#else: 
			#self.node_values[i] = score

		return

	def get_sentence_decision(self, sentence_index):

		current_value = self.node_values[sentence_index]

		return current_value > 0




def get_influence_graph(article, sent_encoder):
	sentences = article['sentences']

	sentence_importances = SummaryRanker().predict_article(article)

	#sentence_encoder = CountVectorizer(max_features=1000, analyzer='char', ngram_range=(2,4))#stop_words='english')
	#sentence_encoder.fit(sentences)
	#sent_encs = np.array(sentence_encoder.transform(sentences).todense())

	#pca = PCA(n_components=3)
	#sent_encs = pca.fit(sent_encs).transform(sent_encs)

	sent_encs = np.array([sent_encoder.encode(s) for s in sentences])

	sentence_similarities = cosine_similarity(sent_encs)

	graph = []

	# compute influence matrix
	for i in range(len(sentences)):

		row = []

		for j in range(len(sentences)):

			correct_order = i < j
			more_important = sentence_importances[i] > sentence_importances[j]
			similarity = sentence_similarities[i][j]
			similarity = (similarity > 0)*similarity
			#similarity = (similarity + 1)/2
			weight = correct_order * more_important * similarity

			row.append(weight)

		graph.append(row)
		#row_str = "\t".join(["{:.2f}".format(v) for v in row])
		#print(row_str)

	return graph