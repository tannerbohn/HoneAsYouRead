
import os
import numpy as np
import itertools
import random
from tqdm import tqdm
import textblob
import pickle
import time


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import keras.backend as K
from keras.models import load_model




from models.GenericSummarizationModel import GenericSummarizationModel
from models.ShowEveryKModel import ShowEveryKModel
from models.ShowFirstModel import ShowFirstModel
from models.LearningModel import LearningModel
from models.ShowAllModel import ShowAllModel
from models.InfluenceGraphModel import InfluenceGraphModel
from models.SkipNextModel import SkipNextModel
from models.SkipNextSimilarModel import SkipNextSimilarModel
from models.SkipAllSimilarModel import SkipAllSimilarModel
from models.OracleModel import OracleModel
from models.CoverageModel import CoverageModel
from models.SimilaritySumSkippingModel import SimilaritySumSkippingModel


from Evaluator import Evaluator
from prepare_dataset import generate_evaluation_data


def print_results(line, results_file):
	print(line)
	results_file.write(line+"\n")
	results_file.flush()

if __name__ == "__main__":

	data_fname = "evaluation_data.pkl"

	if not os.path.exists(data_fname):

		generate_evaluation_data(quick_mode=False, output_fname=data_fname)

	if os.path.exists(data_fname):
		print("Loading data...")
		with open("evaluation_data.pkl", "rb") as f:
			ALL_DATA = pickle.load(f)
		print("Done")
	

	


	train_articles = ALL_DATA["train"]
	validation_articles = ALL_DATA["validation"]
	test_articles = ALL_DATA["test"]#[:100]


	from SentBERTEncoder import SentBERTEncoder
	
	sent_encoder = SentBERTEncoder()

	timestamp = time.strftime("%F-%H:%M:%S")
	results_file = open("results/all_{}.txt".format(timestamp), "w")


	E = Evaluator(sent_encoder)

	'''
	print("Getting baseline performance...")
	model = ShowEveryKModel(k=1)
	print("noisy\tscore")
	for noisy in [True, False]:
		score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)

		print("{}\t{:.3f}".format(noisy, 100*score))

	_ = input("?")
	'''
	baseline = 82.154

	print("Running evaluation...")

	for noisy in [True, False]:
		noise_level = 0.1 if noisy else 0.01
		
		
		model = OracleModel(mode='random')
		score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1, trials=3)
		print_results("noise\tmodel\tscore\tscore_adv\taccept_rate", results_file)
		res_str = "{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, 100*score+baseline, 100*score, accept_rate)
		print_results(res_str, results_file)
		print_results("\n", results_file)

		
		model = OracleModel(mode='sorted')
		score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
		print_results("noise\tmodel\tscore\tscore_adv\taccept_rate", results_file)
		res_str = "{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, 100*score+baseline, 100*score, accept_rate)
		print_results(res_str, results_file)
		print_results("\n", results_file)



		model = OracleModel(mode='greedy', sent_encoder=sent_encoder)
		score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
		print_results("noise\tmodel\tscore\tscore_adv\taccept_rate", results_file)
		res_str = "{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, 100*score+baseline, 100*score, accept_rate)
		print_results(res_str, results_file)
		print_results("\n", results_file)

		'''
		print_results("noise\tmodel\tfrac\tscore\taccept_rate", results_file)
		for frac in [0.25, 0.5, 0.75, 1]:
			model = ShowFirstModel(frac=frac)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}".format(noise_level, model.name, frac, 100*score, accept_rate)
			print_results(res_str, results_file)

		print_results("\n", results_file)
		'''


		print_results("noise\tmodel\tk\tscore\tscore_adv\taccept_rate", results_file)
		for k in [1, 2, 3, 4, 5]:
			model = ShowEveryKModel(k=k)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, k, 100*score+baseline, 100*score, accept_rate)

			print_results(res_str, results_file)

		print_results("\n", results_file)


		print_results("noise\tmodel\tn\tscore\tscore_adv\taccept_rate", results_file)
		for n in [1, 2, 3, 4]:
			model = SkipNextModel(n=n)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, n, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)

		print_results("\n", results_file)


		print_results("noise\tmodel\tthreshold\tscore\tscore_adv\taccept_rate", results_file)
		for similarity_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
			model = SkipNextSimilarModel(sent_encoder=sent_encoder, similarity_threshold=similarity_threshold)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, similarity_threshold, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)

		print_results("\n", results_file)


		print_results("noise\tmodel\tthreshold\tscore\tscore_adv\taccept_rate", results_file)
		for similarity_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
			model = SkipAllSimilarModel(sent_encoder=sent_encoder, similarity_threshold=similarity_threshold)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, similarity_threshold, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)

		print_results("\n", results_file)



		


		print_results("noise\tmodel\tsummarizer\tfrac\tscore\tscore_adv\taccept_rate", results_file)
		for summarizer in ["LexRankSummarizer", "TextRankSummarizer", "SumBasicSummarizer"]:
			for length_frac in [0.25, 0.5, 0.75]:
				model = GenericSummarizationModel(mode='fixed', length_frac=length_frac, summarizer=summarizer)
				score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1)
				res_str = "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, summarizer, length_frac, 100*score+baseline, 100*score, accept_rate)
				print_results(res_str, results_file)

		print_results("\n", results_file)


		print_results("noise\tmodel\tsummarizer\tepsilon\tscore\tscore_adv\taccept_rate", results_file)
		for summarizer in ["LexRankSummarizer", "TextRankSummarizer", "SumBasicSummarizer"]:
			for epsilon in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
				model = GenericSummarizationModel(mode='dynamic', epsilon=epsilon, summarizer=summarizer)
				score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1, trials=3)
				res_str = "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, summarizer, epsilon, 100*score+baseline, 100*score, accept_rate)
				print_results(res_str, results_file)

		print_results("\n", results_file)


		print_results("noise\tmodel\tepsilon\tscore\tscore_adv\taccept_rate", results_file)
		for epsilon in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
			model = LearningModel(sentence_encoder=sent_encoder, mode="constant", epsilon=epsilon, clf_type=LogisticRegression, clf_args={})
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1, trials=3)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, epsilon, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)


		print_results("\n", results_file)


		print_results("noise\tmodel\tbeta\tscore\tscore_adv\taccept_rate", results_file)
		for beta in [0.25, 0.5, 1, 2, 4]:
			model = LearningModel(sentence_encoder=sent_encoder, mode="decreasing", beta=beta, clf_type=LogisticRegression, clf_args={})
			model.beta = beta
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1, trials=3)
			res_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, beta, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)


		print_results("\n", results_file)
		

		print_results("noise\tmodel\tbeta\tc\tscore\tscore_adv\taccept_rate", results_file)
		for beta, c in  itertools.product([0.05, 0.1, 0.25, 0.5, 1, 2, 4], [0, 1, 2, 3, 4, 5]):
		#for beta, c in  itertools.product([16, 32], [8, 9, 10]):
			model = CoverageModel(sent_encoder, n_concepts=4, beta=beta, c=c)
			score, accept_rate = E.evaluate(model, test_articles, noisy=noisy, verbose=1, trials=1)
			res_str = "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(noise_level, model.name, beta, c, 100*score+baseline, 100*score, accept_rate)
			print_results(res_str, results_file)

		print_results("\n====================================\n", results_file)
		

