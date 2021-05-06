
import numpy as np
from types import SimpleNamespace
import nltk

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer

class SummaryRanker:

	def __init__(self, name='TextRankSummarizer'):

		self.stemmer = Stemmer('english')
		self.name = name

		if name == "TextRankSummarizer":
			self.summarizer = TextRankSummarizer(self.stemmer)
		elif name == "LsaSummarizer":
			self.summarizer = LsaSummarizer(self.stemmer)
		elif name == "LuhnSummarizer":
			self.summarizer = LuhnSummarizer(self.stemmer)
		elif name == "LexRankSummarizer":
			setattr(LexRankSummarizer, 'rate_sentences', rate_sentences)
			self.summarizer = LexRankSummarizer(self.stemmer)
			
		elif name == "SumBasicSummarizer":
			self.summarizer = SumBasicSummarizer(self.stemmer)
		elif name == "KLSummarizer":
			self.summarizer = KLSummarizer(self.stemmer)

		#summarizer = EdmundsonSummarizer(stemmer)
		self.summarizer.stop_words = get_stop_words('english')


	def predict_article(self, article):
		sentences = article['sentences']
		summ_sentences, summ_scores = get_summary_scores(sentences, summarizer=self.summarizer)

		#print(summ_scores)

		summ_scores = summ_scores/ np.max(summ_scores)

		if len(sentences)!= len(summ_sentences):
			print("mismatch:", len(sentences), len(summ_sentences))

		return np.array(summ_scores)




# this method is designed to be added to the LexRank summarizer
def rate_sentences(self, document):
	sentences_words = [self._to_words_set(s) for s in document.sentences]
	tf_metrics = self._compute_tf(sentences_words)
	idf_metrics = self._compute_idf(sentences_words)
	matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
	scores = self.power_method(matrix, self.epsilon)
	ratings = dict(zip(document.sentences, scores))
	return ratings





class Tokenizer(object):
	def __init__(self, language):
		self.language = language
		self._sentence_tokenizer = lambda text: text.split("<SENTENCE>")
		self._word_tokenizer = lambda text: nltk.word_tokenize(text)

	def to_sentences(self, text):
		return self._sentence_tokenizer(text)

	def to_words(self, text):
		return self._word_tokenizer(text)


def get_summary_scores(sentences, summarizer):

	text = '<SENTENCE>'.join(sentences)
	parser = PlaintextParser.from_string(text, Tokenizer('english'))

	try:
		rating_dict = summarizer.rate_sentences(parser.document)
	except:
		rating_dict = summarizer._compute_ratings(parser.document.sentences)
	
	ratings = np.array([rating_dict[s] for s in parser.document.sentences])


	m = np.min(ratings)
	M = np.max(ratings)
	if m == M:
		ratings = np.ones(len(ratings))
	else:
		ratings = (ratings - m)/(M - m)

	summ_sentences = [str(s) for s in parser.document.sentences]
	return summ_sentences, ratings
