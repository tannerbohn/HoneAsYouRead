
from tqdm import tqdm
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

class Evaluator:

	def __init__(self, encoder):

		self.encoder = encoder

		return


	def evaluate(self, model, articles, noisy=False, verbose=1, trials=1):

		if trials > 1:

			trial_scores = []
			trial_swipes = []

			for t in range(trials):
				score, swipes = self.evaluate(model, articles, noisy, verbose, trials=1)
				trial_scores.append(score)
				trial_swipes.append(swipes)

			print("score stddev:", np.std(trial_scores))
			print("swipes stddev:", np.std(trial_swipes))

			return np.average(trial_scores), np.average(trial_swipes)



		scores = []

		reject_rates = []

		accept_rates = []

		#time_saved = []

		noise_level = 0.1 if noisy else 0.01

		for article in tqdm(articles, disable=verbose==0):

			#print("PREPARING FOR ARTICLE")
			model.prepare_for_article(article)

			produced_summary = []

			sentences = article['sentences']
			user_feedbacks = article['feedbacks'][noise_level]
			length_limit = article['length_limit']

			nb_rejected = 0
			nb_accepted = 0

			nb_sents_shown = 0


			for i_s in range(len(sentences)):

				if model.get_decision(i_s):

					nb_sents_shown += 1
		
					model.incorporate_sample(i_s, user_feedbacks[i_s], True)

					# since we are using negative feedback, 1 = no swipe, 0 = swipe
					accepted = user_feedbacks[i_s]
					if accepted:
						nb_accepted += 1
					else:
						nb_rejected += 1


					produced_summary.append(sentences[i_s])

					if verbose >= 2:
						print("SHOW\t{}\t{}".format(i_s, sentences[i_s][:100]))
						print("\tfeedback: ", user_feedbacks[i_s])

					#if len(produced_summary) >= length_limit:
					if nb_sents_shown >= length_limit:
						if verbose >= 2:
							print("----- STOPPED READING -----")
							for i_missed in range(i_s+1, len(sentences)):
								print("UNREAD\t{}\t{}".format(i_missed, sentences[i_missed][:100]))
								print("\tmissed feedback:", user_feedbacks[i_missed])

						break
				else:
					model.incorporate_sample(i_s, None, False)
					if verbose >= 2:
						print("HIDE\t{}\t{}".format(i_s, sentences[i_s][:100]))
						print("\tmissed feedback:", user_feedbacks[i_s])
					
			#print("summary len:", len(produced_summary))

			score = self.score_summary(produced_summary, sentences, article['importances'], article['length_limit'])

			scores.append(score)

			reject_rates.append(nb_rejected/nb_sents_shown)
			accept_rates.append(nb_accepted/nb_sents_shown)

			#time_saved.append(length_limit - nb_sents_shown)

		reject_rate = np.average(reject_rates)
		accept_rate = np.average(accept_rates)

		return np.average(scores), accept_rate #, np.average(time_saved)

	def score_summary(self, summary_sentences, article_sentences, true_sentence_weights, length_limit):

		if len(summary_sentences) == 0: return 0

		# encode summary sentences
		summ_vec_list = [self.encoder.encode(s) for s in summary_sentences]

		# encode article sentences
		doc_sent_vecs = [self.encoder.encode(s) for s in article_sentences]

		sim = get_sim_metric(summ_vec_list, doc_sent_vecs, true_sentence_weights, method='w')

		#unread_sent_vecs = [v for s, v in zip(article_sentences, doc_sent_vecs) if s not in summary_sentences]
		#unread_sim = 0 if len(unread_sent_vecs) == 0 else get_sim_metric(unread_sent_vecs, doc_sent_vecs, true_sentence_weights, method='w')

		baseline_sim = get_sim_metric(doc_sent_vecs[:length_limit], doc_sent_vecs, true_sentence_weights, method='w')


		return sim - baseline_sim # - unread_sim # 


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

		#sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))

		#return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)
		dist_matrix = cosine_distances(np.array(ref_vecs),np.array(summ_vec_list))

		return 1 - np.dot(np.array(np.min(dist_matrix,axis=1)),np.array(weights))/np.sum(weights)


