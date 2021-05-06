
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

class OracleModel:

	def __init__(self, mode, sent_encoder=None):

		# mode: either "feedback" or "sorted" or "greedy" or "random"

		self.name = "oracle_{}".format(mode)

		self.mode = mode

		self.sent_encoder = sent_encoder

		return

	def prepare_for_article(self, article):

		self.article = article

		self.decisions = []

		if self.mode == "sorted":

			importances = article['importances']
			index_scores = zip(range(len(importances)), importances)
			index_scores = sorted(index_scores, key = lambda k: -k[1])

			indices, _ = list(zip(*index_scores))

			top_indices = indices[:article['length_limit']]

			for i in range(len(article['sentences'])):
				self.decisions.append(i in top_indices)

		elif self.mode == "random":

			importances = [random.random() for _ in article['sentences']]
			index_scores = zip(range(len(importances)), importances)
			index_scores = sorted(index_scores, key = lambda k: -k[1])

			indices, _ = list(zip(*index_scores))

			top_indices = indices[:article['length_limit']]

			for i in range(len(article['sentences'])):
				self.decisions.append(i in top_indices)

		elif self.mode == "greedy":

			if "greedy_oracle_decision" in article:
				self.decisions = article["greedy_oracle_decision"]
			else:

				sentences = article['sentences']

				self.decisions = [False for _ in sentences]


				chosen_sents = []

				# instead of greedily adding sentences by going through the normal order, 
				# first sort by importance -- should be faster
				importances = article['importances']
				index_scores = zip(range(len(importances)), importances)
				index_scores = sorted(index_scores, key = lambda k: -k[1])

				sorted_indices, _ = list(zip(*index_scores))

				remaining_sent_indices = list(sorted_indices) #list(range(len(sentences))) #[s for s in article['sentences']]

				while len(chosen_sents) < article['length_limit']:

					best_score = 0
					best_index = None

					for i in remaining_sent_indices:

						summary = chosen_sents+[sentences[i]]
						new_score = score_summary(self.sent_encoder, summary, sentences, article['importances'])

						if new_score > best_score:
							best_score = new_score
							best_index = i

					chosen_sents.append(sentences[best_index])
					remaining_sent_indices.remove(best_index)
					self.decisions[best_index] = True

		return

	def get_decision(self, sentence_index):

		return self.decisions[sentence_index]

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		return



def score_summary(sent_encoder, summary_sentences, article_sentences, true_sentence_weights):

	# encode summary sentences
	summ_vec_list = [sent_encoder.encode(s) for s in summary_sentences]

	# encode article sentences
	doc_sent_vecs = [sent_encoder.encode(s) for s in article_sentences]

	sim = get_sim_metric(summ_vec_list, doc_sent_vecs, true_sentence_weights, method='w')


	return sim


def get_sim_metric(summ_vec_list, doc_sent_vecs, doc_sent_weights, method='cos'):
	# from https://github.com/yg211/acl20-ref-free-eval/blob/tac_summarisation/ref_free_metrics/utils.py#L36

	#print('weights', doc_sent_weights)
	# get the avg doc vec, then cosine
	if method == 'cos':
		summ_vec = np.mean(np.array(summ_vec_list),axis=0)
		dvec = np.matmul(np.array(doc_sent_weights).reshape(1,-1),  np.array(doc_sent_vecs))
		return cosine_similarity(dvec,summ_vec.reshape(1,-1))[0][0]

	# bert-score, quicker to run and gives similar performance to mover-bert-score
	elif method == 'w':
		#ref_vecs = [doc_sent_vecs[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]

		#weights = [doc_sent_weights[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
		ref_vecs = doc_sent_vecs

		weights = doc_sent_weights

		sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))

		return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)