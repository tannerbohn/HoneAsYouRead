


import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm




class SentBERTEncoder:

	def __init__(self, precomputed_embeddings=None):

		

		# https://arxiv.org/pdf/1908.10084.pdf
		# https://github.com/UKPLab/sentence-transformers

		'''
		"roberta-base-nli-stsb-mean-tokens" - 768 dim
		"bert-base-nli-stsb-mean-tokens" - 768
		"bert-base-nli-max-tokens" 
		"bert-base-nli-mean-tokens"
		'''
		from sentence_transformers import SentenceTransformer

		self.name = "sentbert"

		self.shape = (768,)

		self.sent_encoder = SentenceTransformer("bert-base-nli-mean-tokens")

		self._precomputed_sent_encs = dict()

		# be able to use a subset of the embedding dimensions
		self.dimensions = "all"

		if precomputed_embeddings != None:
			print("loading precomputed embeddings...")
			self.load_precomputed_embeddings(precomputed_embeddings)
			print("done")

		return

	def load_precomputed_embeddings(self, filename):

		filenames = filename if type(filename) == list else [filename]

		for fname in filenames:
			print("Loading embeddings from ", fname, "...")
			with open(fname, "rb") as f:
				self._precomputed_sent_encs.update(pickle.load(f))
		print("DONE")



	def set_dimensions(self, new_dimensions):
		if type(new_dimensions) == int:
			self.dimensions = (new_dimensions,)
		elif type(new_dimensions) == list:
			self.dimensions = tuple(new_dimensions)
		elif new_dimensions == "all":
			self.dimensions = "all"


	def reset(self):
		self._precomputed_sent_encs = dict()

	def precompute(self, sentences):

		vecs = self.sent_encoder.encode(sentences, show_progress_bar=True)
		for i_s, sentence in enumerate(sentences):
			self._precomputed_sent_encs[sentence] = vecs[i_s]


	def encode(self, text, document=None):

		#if type(text) == list:
		#	encs = [self.encode(sentence) for sentence in text]
		#	return np.average(encs, axis=0)

		sentence = text


		try:
			enc = self._precomputed_sent_encs[sentence]

			if self.dimensions == "all":
				return enc
			elif len(self.dimensions) == 1:
				return np.array([enc[self.dimensions]])
			else:
				return enc[self.dimensions]
		except:

			enc = self.sent_encoder.encode([sentence], show_progress_bar=False)[0]
			self._precomputed_sent_encs[sentence] = enc

			if self.dimensions == "all":
				return enc
			elif len(self.dimensions) == 1:
				return np.array([enc[self.dimensions]])
			else:
				return enc[self.dimensions]

	def batch_encode(self, texts, verbose=0):

		vecs = self.sent_encoder.encode(texts, show_progress_bar=verbose>0)
		if self.dimensions == "all":
			return vecs
		else:
			return vecs[:,self.dimensions]
