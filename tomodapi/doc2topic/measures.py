# Evaluation metrics from old Keras code base

import numpy as np
import requests
import tensorflow.keras.backend as K


def precision(y_true, y_pred):
	'''Calculates the precision, a metric for multi-label classification of
	how many selected items are relevant.
	'''
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	'''Calculates the recall, a metric for multi-label classification of
	how many relevant items are selected.
	'''
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def fbeta_score(y_true, y_pred, beta=1):
	'''Calculates the F score, the weighted harmonic mean of precision and recall.

	This is useful for multi-label classification, where input samples can be
	classified as sets of labels. By only using accuracy (precision) a model
	would achieve a perfect score by simply assigning every class to every
	input. In order to avoid this, a metric should penalize incorrect class
	assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
	computes this, as a weighted mean of the proportion of correct class
	assignments vs. the proportion of incorrect class assignments.

	With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
	correct classes becomes more important, and with beta > 1 the metric is
	instead weighted towards penalizing incorrect class assignments.
	'''
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score


def fmeasure(y_true, y_pred):
	'''Calculates the f-measure, the harmonic mean of precision and recall.
	'''
	return fbeta_score(y_true, y_pred, beta=1)


# Custom metrics

L2 = (lambda x: np.linalg.norm(x, 2))
L1 = (lambda x: np.linalg.norm(x, 1))
L1normalize = (lambda x: x/L1(x))
relufy = np.vectorize(lambda x: max(0., x))


def sparsity(vecs, n=-1):
	""" Distribution sparsity measured by L2norm/L1norm """
	return np.nanmean([L2(x)/L1(x) for x in vecs[:n,]])


def peak_rate(vecs, factor, n=-1):
	""" Rate of dimensions with values above the threshold factor/number_of_dimensions """
	return np.mean([sum(dist/L1(dist) > factor/len(dist)) for dist in vecs[:n,]])/vecs.shape[1]


def topic_overlap(topic_words):
	""" Measure word overlap between top words for topics.
		Maximum overlap between one topic and the rest is calculated and averaged over all topics. """
	overlaps = []
	for topic_i in topic_words:
		max_overlap = 0
		topic_bow1 = set([word for word, _ in topic_words[topic_i]])
		for topic_j in topic_words:
			if topic_i == topic_j:
				continue
			topic_bow2 = set([word for word, _ in topic_words[topic_j]])
			max_overlap = max(max_overlap, len(topic_bow1.intersection(topic_bow2)))
		overlaps.append(max_overlap/len(topic_words[0]))
	return np.mean(overlaps)


def topic_prec_recall(topic_words, idx2token, counter, n_freq_words=100, stopwords=set()):
	""" Measure fraction of n_freq_words covered by topic_words """
	all_topic_words = set()
	for topic in topic_words:
		all_topic_words |= set([word for word, _ in topic_words[topic]])
	#eval_size = len(topic_words[0])*wordvecs.shape[1]
	top_words = sorted([(cnt, word) for word, cnt in counter.items()])[-2*n_freq_words:]
	top_words = set([word for word, _ in top_words if word not in stopwords][-1*n_freq_words:])
	recall = len(all_topic_words.intersection(top_words))/len(top_words)
	prec = len(all_topic_words.intersection(top_words))/len(all_topic_words)
	return prec, recall


def cv_coherence(topic_words):
	""" C_v topic coherence measure (API) """
	coherences = []
	print("Calling topic coherence API...\nTopic\tCoherence")
	for topic in topic_words:
		word_list = '%20'.join([word.replace('#','').replace('+','%2B') for word, _ in topic_words[topic]][:10])
		page = requests.get("http://palmetto.aksw.org/palmetto-webapp/service/cv?words=%s" % word_list)
		if page.status_code != 200:
			print("Error: Received HTTP code", page.status_code)
			continue
		if page:
			coherences.append(float(page.text))
			print("%d\t%.5f" % (topic, coherences[-1]))
		else:
			print("Error: Could not read page")
			continue
	print("Mean\t%.5f" % np.mean(coherences))
	return np.mean(coherences)


def pmix(word1, word2, counter, cocounter, blacklist=set()):
	""" Topic coherence based on point-wise mutual information """
	w1, w2 = sorted([word1, word2])
	if not w1.replace('#','').replace('-','').isalpha() or not w1.replace('#','').replace('-','').isalpha():
		return 0
	if w1 in blacklist or w2 in blacklist:
		return 0
	try:
		return max(0, np.log(cocounter[w1][w2]/((counter[word1]+counter[word2])/sum(counter.values()))))
	except KeyError:
		return np.nan


def pmix_coherence(topic_top_words, counter, cocounter, blacklist=set()):
	""" Aggregate PMIx scores """
	return np.nanmean([
		np.nanmean([pmix(word1, word2, counter, cocounter, blacklist=blacklist) for word2 in topic_top_words if word1 != word2])
		for word1 in topic_top_words
	])


def topic_wordiness(topic_words):
	cnt, tot = 0, 0.
	for topic in topic_words:
		words = [word.replace('#','').replace('-','') for word, _ in topic_words[topic]][:10]
		cnt += len([word for word in words if word.isalpha()])
		tot += len(words)
	return cnt/tot


def topic_stopwordiness(topic_words, stopwords):
	cnt, tot = 0, 0.
	for topic in topic_words:
		words = [idx2token[word].replace('#','') for word, _ in topic_words[topic]][:10]
		cnt += len([word for word in words if word in stopwords])
		tot += len(words)
	return cnt/tot
