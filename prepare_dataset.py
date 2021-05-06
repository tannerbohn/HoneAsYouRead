
import numpy as np
import itertools
import random
from tqdm import tqdm
import textblob
import pickle
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, cosine_distances
from sklearn.decomposition import PCA


from models.OracleModel import OracleModel


def FP(r, alpha, m):
	return 1/(1 + np.exp((alpha - r)/m))

def stochastic_feedback_transformation(values, length, feedback_type, m=0.1):

	#m: noisiness

	lenfrac = length/len(values)
	# connect alpha to length limit: long limit = low alpha_positive
	#alpha_positive = 1 - length/len(values)

	
	

	feedbacks = []

	if feedback_type == "positive":

		# reaction threshold
		alpha = 0.75 #alpha_positive

		for r in values:
			p = FP(r, alpha, m)
			f = random.random() < p
			#print("{:.3f}\t{:.3f}\t{}".format(r, p, f))
			feedbacks.append(f)

	elif feedback_type == "negative":

		# reaction threshold
		# when lenfrac is small, alpha should be large (so that more sentences are swiped away)
		# when lenfrac is large, alpha should be small (so that only the worst sents
		# 	are swiped away)
		alpha = 1 - lenfrac #1 - alpha_positive

		for r in values:
			p = 1 - FP(r, alpha, m)
			f = random.random() < p
			#print("{:.3f}\t{:.3f}\t{}".format(r, p, f))
			# make it so that swipe = 0, not swipe = 1
			feedbacks.append(1-f)

	return feedbacks
		

def compute_concept_vectors(sentences, sent_encoder, n=4):

	#sentence_encoder = CountVectorizer(max_features=1000, analyzer='char', ngram_range=(2,4))#stop_words='english')
	#sentence_encoder.fit(sentences)

	#sent_encs = np.array(sentence_encoder.transform(sentences).todense())

	

	sent_encs = np.array([sent_encoder.encode(s) for s in sentences]) #sent_encoder.batch_encode(sentences) #

	#pca = PCA(n_components=5)
	X_dim_reduced = sent_encs #pca.fit(sent_encs).transform(sent_encs)

	'''
	distances = euclidean_distances(X_dim_reduced, X_dim_reduced).flatten()
	d_min = np.min(distances)
	d_10p = np.percentile(distances, 10)
	d_25p = np.percentile(distances, 25)
	d_50p = np.percentile(distances, 50)
	d_90p = np.percentile(distances, 90)
	d_max = np.max(distances)
	print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(d_min, d_10p, d_50p, d_90p, d_max))
	'''

	cluster = KMeans(n_clusters=n, random_state=0).fit(X_dim_reduced)
	
	#cluster = AffinityPropagation(max_iter = 500).fit(X_dim_reduced)
	#cluster = MeanShift().fit(X_dim_reduced)
	#cluster = DBSCAN(min_samples = 3, eps=d_10p, metric='cosine').fit(X_dim_reduced)
	#cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=d_25p).fit(X_dim_reduced)

	#cluster = KMeans(n_clusters=n, random_state=0).fit(sent_encs)
	centers = cluster.cluster_centers_ #cluster.components_ #
	
	#print("{}\t{}".format(len(centers), "" if len(centers) > 0 else "FAILED"))
	

	#print("\n\n")
	#for i, xs in enumerate(X_dim_reduced):
	#	print("{}\t{:.4f}\t{:.4f}\t{}".format(cluster.labels_[i], xs[0], xs[1], sentences[i].replace("\"", '\'')))


	
	return centers, X_dim_reduced



def process_article(article, sent_encoder, nb_concepts=4, noisiness_options=[0.01, 0.1]):

	sentences = article['sentences']

	concept_vectors, sent_vecs = compute_concept_vectors(sentences, sent_encoder, n=nb_concepts)


	concept_weights = np.random.uniform(0, 1, nb_concepts)
	concept_weights = concept_weights/np.max(concept_weights)


	sim_matrix = cosine_similarity(sent_vecs, concept_vectors)

	importances = np.max(sim_matrix * concept_weights,axis=1)

	#print("Concept weights:", concept_weights)
	#for i_s, sent in enumerate(sentences):
	#	print("{}\t{:.2f}\t{}".format(np.around(sim_matrix[i_s], decimals=2), importances[i_s], sent))
	#print()


	article['importances'] = importances
	#print(sorted(importances))
	
	'''
	min_i = np.min(importances)
	max_i = np.max(importances)
	p50 = np.percentile(importances, 50)
	p90 = np.percentile(importances, 90)
	p10 = np.percentile(importances, 10)
	print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(min_i, p10, p50, p90, max_i))
	'''	

	# TODO: make length preference uniformly distributed? (to represent a wider array of preferences)
	# generate length preference

	length_limit = np.random.randint(1, len(sentences)+1) #len(sentences)//2 #
	article['feedbacks'] = dict()
	for noisiness in [0.01, 0.1]:
		article['feedbacks'][noisiness] = stochastic_feedback_transformation(values=importances, length = length_limit, feedback_type="negative", m=noisiness)

	article['length_limit'] = length_limit

	oracle_model = OracleModel(mode='greedy', sent_encoder=sent_encoder)
	oracle_model.prepare_for_article(article)

	article['greedy_oracle_decision'] = oracle_model.decisions


def get_sentences(text):
	return [str(s) for s in textblob.TextBlob(text).sentences]


def generate_evaluation_data(quick_mode=False, output_fname=None):
	'''
	if output_fname == None, returns data
	else: print it to that file

	'''

	import nlp
	#from gensim.summarization.textcleaner import split_sentences

	from SentBERTEncoder import SentBERTEncoder
	sent_encoder = SentBERTEncoder()

	print("Loading CNN DM dataset...")
	cnndm_dataset = nlp.load_dataset('cnn_dailymail', '3.0.0')

	#articles = load_articles()

	# the pyarrow data type is a little unusual...
	print("Preparing dataset...")
	#articles = [{"sentences": get_sentences(text)} for text in cnndm_dataset['train'][:500]['article']]

	np.random.seed(123)
	random.seed(123)


	nb_sampes_per_partition = 100 if quick_mode else 30000

	data = dict()

	start_time = time.time()
	for partition in ['test', 'validation', 'train']:

		data[partition] = []

		article_IDs = cnndm_dataset[partition][:nb_sampes_per_partition]['id']
		nb_articles = len(article_IDs)
		print("articles:", nb_articles)
		for i, (text, article_id) in enumerate(zip(cnndm_dataset[partition][:nb_sampes_per_partition]['article'], article_IDs)):
			if i % 10 == 0:
				cur_time = time.time()
				seconds_elapsed = cur_time - start_time
				total_est_time = seconds_elapsed/((i+1) / nb_articles)
				seconds_remaining = total_est_time - seconds_elapsed
				hours_remaining = seconds_remaining/3600
				print("{}\t{}\t{:.4f}%\t{:.2f} hours".format(partition, i, 100*i/nb_articles, hours_remaining))
			sentences = get_sentences(text)
			if len(sentences) < 10: continue
			# get importances, feedback, length limit, greedy_oracle decisions
			article = {"sentences": sentences, "id":article_id}
			process_article(article=article, sent_encoder=sent_encoder, nb_concepts=4, noisiness_options=[0.01, 0.1])
			#del article['sentences']
			sent_encoder.reset() # to save memory -- since articles are unlikely to repeat stuff very often
			#data[article_id] = article
			data[partition].append(article)



	if output_fname == None:
		print("No output file specified, returning data...")
		return data
	else:
		print(f"Saving to file ({output_fname})...")
		with open(output_fname, "wb") as f:
			pickle.dump(data, f)
