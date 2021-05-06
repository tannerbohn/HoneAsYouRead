
import numpy as np


# simply copy the observed label of the previous sentence
class PrevModel:

	def __init__(self, window=1):

		# window is the size of history to average over
		self.window = window

		return

	def iterative_predict(self, article):

		#if mode == "realistic": return self.iterative_predict_realistic(article)

		predicted_labels = []
		observed_labels = []

		sentences = article['sentences']
		nb_sents = len(sentences)

		for i_s in range(nb_sents):

			if len(observed_labels) == 0:

				predicted_labels.append(1)

			else:

				predicted_labels.append(np.average(observed_labels[-self.window:]))
				#predicted_labels.append(observed_labels[-1])

			observed_labels.append(article['labels'][i_s])

		return np.array(predicted_labels)
