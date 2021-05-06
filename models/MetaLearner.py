from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model, clone_model
from keras.activations import softmax

from keras import backend as K

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import random
import math

class MetaLearner:

	def __init__(self, sent_encoder, model=None):

		self.name = "metalearner"

		self.sent_encoder = sent_encoder

		self.model = model
		return

	def train(self, evaluator, train_articles, population_size=6, generations=20, noisy=True, verbose=False):

		input_shape = (7+3,)

		K.clear_session()

		best_weights = None
		best_score = -float('inf')

		rate = 0.3
		decay = 0.99

		f = open("results.txt", "w")

		prev_scores = None

		history_length = 4
		score_history = []
		weight_history = []

		consecutive_worse = 0
		consecutive_improve = 0

		for gen in range(generations):
			#print("Generation", gen)


			#print(best_weights)

			if gen == 0:
				W_new = create_model(input_shape=input_shape).get_weights()
				
			elif gen == 1:
				W_new = mutate_weights(weights=weight_history[-1], rate=rate, sparsity=0.9)
			#elif len(score_history) > 2 and score_history[-1] < best_score - 1 and random.random() <0.2 and consecutive_improve < 2:
			#	print("Going back to good solution...")
			#	W_new = mutate_weights(weights=best_weights, rate=rate, sparsity=0.9)

			#elif len(score_history) == history_length and (max(score_history) - min(score_history)) <= 0.001:
			#	print("Starting from random solution...")
			#	W_new = create_model(input_shape=input_shape).get_weights()

			else:

				score_delta = score_history[-1] - score_history[-2]

				W_a = weight_history[-2]
				W_b = weight_history[-1]

				#alpha = min(0.25, abs(score_delta)**2) #min(0.1, abs(score_delta))
				alpha = min(0.25, abs(score_delta)) #min(0.1, abs(score_delta))

				#if score_delta > 0:
				W_new = linearly_combine_weights([W_b, W_a], [1+alpha, -alpha])
				#W_new = linearly_combine_weights([W_a, W_b], [1-alpha, alpha])
				W_new = mutate_weights(weights=W_new, rate=max(rate, alpha), sparsity=0.9)
				#else:
				#	W_new = linearly_combine_weights([W_b, W_a], [1+0.2, -0.2])

			weight_history.append(W_new)



			model = MetaLearner(sent_encoder=self.sent_encoder, model=create_model(input_shape, weights=W_new))


			article_indices = list(range(len(train_articles)))
			random.shuffle(article_indices)
			if gen < 100:
				article_indices = article_indices[:100]
			else:
				article_indices = article_indices[:1000]#min(1000, 50*(gen+1))]#(gen+1)*50]
			eval_articles = [train_articles[i] for i in article_indices]

			#eval_articles = train_articles

			score, swipes = evaluator.evaluate(model, eval_articles, noisy=noisy, verbose=False)
			score *= 100
			res_str = "{}\t{:.3f}\t{:.3f}".format(gen, rate, score)
			print(res_str)
			f.write(res_str+"\n")
			f.flush()

			score_history.append(score)

			weight_history = weight_history[-history_length:]
			score_history = score_history[-history_length:]

			if score > best_score:
				best_score = score

				best_weights = [w.copy() for w in W_new]
				consecutive_worse = 0
				consecutive_improve += 1
			else:
				consecutive_worse += 1
				consecutive_improve = 0

			rate = rate * decay

			K.clear_session()

			self.model = create_model(input_shape, weights=best_weights)

		f.close()

		#self.model = create_model(input_shape, weights=best_weights)

		return

	def train_orig(self, evaluator, train_articles, population_size=6, generations=20, noisy=True, verbose=False):

		input_shape = (7,)

		K.clear_session()

		best_weights = None
		best_score = -float('inf')

		rate = 0.5
		decay = 0.95

		f = open("results.txt", "w")

		prev_scores = None

		for gen in range(generations):
			#print("Generation", gen)

			#print(best_weights)

			if gen == 0:
				population = [create_model(input_shape=input_shape).get_weights() for _ in range(population_size)]
			else:

				avg_score = np.average(prev_scores)
				good_weights = [W for i, W in enumerate(population) if prev_scores[i] >= avg_score]
				bad_weights = [W for i, W in enumerate(population) if prev_scores[i] < avg_score]
				W_good_avg = get_weight_average(good_weights)
				W_bad_avg = get_weight_average(bad_weights)

				#W_delta = linearly_combine_weights([W_good_avg, W_bad_avg], [1, -1])
				#W_better = linearly_combine_weights([W_good_avg, W_delta], [1, 0.5])

				#W_better = linearly_combine_weights([W_good_avg, W_bad_avg], [1+0.25, -0.25])

				W_good_estimates = [linearly_combine_weights([W_good_avg, W_bad_avg], [1+gamma, -gamma]) for gamma in [0, 0.1, 0.2, 0.5]]
				W_best_estimate = linearly_combine_weights([best_weights, W_bad_avg], [1+0.2, -0.2])

				population = [W_best_estimate]+[mutate_weights(weights=random.choice(W_good_estimates), rate=rate, sparsity=0.9) for _ in range(population_size - 1)]

			models = [MetaLearner(sent_encoder=self.sent_encoder, model=create_model(input_shape, weights=W)) for W in population]

			scores = []
			for m in models:
				score, swipes = evaluator.evaluate(m, train_articles, noisy=noisy, verbose=False)
				#print("\tscore:", score)
				res_str = "{}\t{:.3f}\t{:.3f}".format(gen, rate, 100*score)
				print(res_str)
				f.write(res_str+"\n")
				f.flush()
				scores.append(score)

			gen_top_score = max(scores)

			if gen_top_score > best_score:
				best_score = gen_top_score

				best_index = np.argmax(scores)

				best_weights = population[best_index]

			rate = rate * decay
			prev_scores = scores

			K.clear_session()

			self.model = create_model(input_shape, weights=best_weights)

		f.close()

		#self.model = create_model(input_shape, weights=best_weights)

		return

	def prepare_for_article(self, article):

		self.sentences = article['sentences']
		self.nb_sents = len(article['sentences'])

		self.sent_encs = [self.sent_encoder.encode(s) for s in self.sentences]

		self.disliked_sent_encs = []
		self.liked_sent_encs = []
		self.prev_status = 0

		self.mem_vec = np.zeros(3)

		return

	def get_decision(self, sentence_index):

		frac_through = (sentence_index+1)/self.nb_sents

		sent_enc = self.sent_encs[sentence_index]

		if len(self.disliked_sent_encs) == 0:
			max_sim_to_disliked = 0
			sim_to_avg_disliked = 0
		else:
			sim_to_disliked = cosine_similarity([sent_enc], self.disliked_sent_encs)
			max_sim_to_disliked = np.max(sim_to_disliked)
			sim_to_avg_disliked = np.average(sim_to_disliked)


		if len(self.liked_sent_encs) == 0:
			max_sim_to_liked = 0
			sim_to_avg_liked = 0
		else:
			sim_to_liked = cosine_similarity([sent_enc], self.liked_sent_encs)
			max_sim_to_liked = np.max(sim_to_liked)
			sim_to_avg_liked = np.average(sim_to_liked)

		frac_liked = 1 if sentence_index == 0 else len(self.liked_sent_encs)/sentence_index

		# log doc length
		# log position
		# similarity to doc avg
		# memory

		#log_doc_len = math.log10(self.nb_sents)
		#log_pos = math.log10(sentence_index+1)

		# features needed for coverage:
		#	- current coverage
		#	- estimate of concept importances
		#	- concepts of current sentence

		feature_vec = [frac_through, max_sim_to_disliked, sim_to_avg_disliked, max_sim_to_liked, sim_to_avg_liked, frac_liked, self.prev_status]+list(self.mem_vec)#, log_doc_len, log_pos]

		output = self.model.predict(np.array([feature_vec]))[0]
		show_prob = output[0]
		self.mem_vec = output[1:]
		#print(show_prob)#, random.random() < show_prob)

		#print(show_prob, self.mem_vec)

		return random.random() < show_prob

	def incorporate_sample(self, sentence_index, feedback, is_observed):

		if not is_observed:
			self.prev_status = 0#"hidden"
			return

		if feedback == 0:
			self.disliked_sent_encs.append(self.sent_encs[sentence_index])
			self.prev_status = -1 #"disliked"
		else:
			self.liked_sent_encs.append(self.sent_encs[sentence_index])
			self.prev_status = 1 #"liked"

		return

def create_model(input_shape, weights=None):

	main_input = Input(shape = input_shape)

	hl_1 = Dense(32, activation="relu")(main_input)
	hl_2 = Dense(32, activation="relu")(hl_1)

	output = Concatenate()([Dense(1, activation="sigmoid")(hl_2), Dense(3, activation="tanh")(hl_2)])
	#output = Dense(1, activation="sigmoid")(hl_2)

	model = Model(main_input, output)

	#print("type:", type(weights))
	if type(weights) == list: #'numpy.ndarray':
		model.set_weights(weights)

	model.compile(optimizer='adam', loss='binary_crossentropy')


	#if type(weights) == list:
	#	actual_weights = model.get_weights()
	#	print(actual_weights[0] == weights[0])

	


	return model


def mutate_weights(weights, rate=0.5, sparsity=0.9):

	W_orig = weights #model.get_weights

	W_new =[]

	for w in W_orig:

		shape = w.shape

		noise = np.random.normal(0, rate, size=shape) #np.random.uniform(-rate, rate, size=shape) #
		mask = np.random.choice([0, 1], p=[1-sparsity, sparsity], size=shape)
		w_new = w + noise*mask

		W_new.append(w_new)

	#new_model = clone_model(model)
	#new_model.set_weights(W_new)

	#return new_model
	return W_new

def get_weight_average(W_list):

	W_new = []

	n = len(W_list[0])

	for l in range(n):

		w = np.average([W[l] for W in W_list], axis=0)

		W_new.append(w)

	return W_new

def linearly_combine_weights(W_list, multipliers):

	W_new = []

	n = len(W_list[0])

	for l in range(n):

		w = np.sum([W[l]*multipliers[i] for i, W in enumerate(W_list)], axis=0)

		W_new.append(w)

	return W_new