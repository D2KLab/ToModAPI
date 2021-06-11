import numpy as np
import os
import pickle
import gensim
import logging
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics

from gensim.test import utils
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class AbstractModel:
    ROOT = '.'

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

        self.log = logging.getLogger(self.__class__.__name__)

    def load(self, path=None):
        """
            Load the model and eventual dependencies.

            :param path: Folder where the model to be loaded is. If not specified, a default one is assigned
        """
        #  Implementation not mandatory.
        if path is not None:
            self.model_path = path

    def save(self, path=None):
        """
            Save the model and eventual dependencies.

            :param path: Folder where to save the model. If not specified, a default one is assigned
        """
        if path is not None:
            self.model_path = path

    # Perform Inference
    def predict(self, text, topn=5, preprocessing=False):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocess: If True, execute preprocessing on the document
        """
        raise NotImplementedError

    def predict_corpus(self, datapath=ROOT + '/data/test.txt', topn=5):
        if self.model is None:
            self.load()

        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        return [self.predict(t, topn=topn) for t in text]

    def train(self, data=ROOT + '/data/test.txt', num_topics=20, preprocessing=False):
        """ Train topic model.

            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
        """
        raise NotImplementedError

    @property
    def topics(self):
        """ List of the topics computed by the model

            :returns: a list of topic objects containing
            - 'words' the list of words related to the topic
            - 'weights' of those words in order (not always present)
        """
        raise NotImplementedError

    def topic(self, topic_id: int):
        """ Get info on a given topic

            :param int topic_id: Id of the topic
            :returns: an object containing
            - 'words' the list of words related to the topic
            - 'weights' of those words in order (not always present)
        """

        return self.topics[topic_id]

    def get_corpus_predictions(self, topn=5):
        """
        Returns the predictions computed on the training corpus.
        This is not re-computing predictions, but reading training results.

        :param int topn: Number of most probable topics to return for each document
        """
        raise NotImplementedError

    def coherence(self, datapath=ROOT + '/data/test.txt', metric='c_v', glove_path='glove/glove.6B.300d.txt'):
        """ Get the coherence of the topic mode.

        :param datapath: Path of the corpus on which compute the coherence.
        :param metric: Metric for computing the coherence, among <c_v, c_npmi, c_uci, u_mass, c_we>
         """
        if metric not in ['c_v', 'c_npmi', 'c_uci', 'u_mass', 'c_we']:
            raise RuntimeError('Unrecognised metric: ' + metric)

        topic_words = [x['words'] for x in self.topics]

        self.log.debug('loading dataset')
        with open(datapath, "r") as datafile:
            text = [line.rstrip().split() for line in datafile if line]

        dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

        results = {}

        if metric == 'c_we':

            if os.path.exists(glove_path.replace('txt', 'pickle')):
                glove = pickle.load(open(glove_path.replace('txt', 'pickle'), 'rb'))
            else:
                w2v = utils.get_tmpfile("w2v")
                glove2word2vec(glove_path, w2v)
                glove = KeyedVectors.load_word2vec_format(w2v)
                pickle.dump(glove, open(glove_path.replace('txt', 'pickle'), 'wb'))

            results['c_we_per_topic'] = []

            for topic in topic_words:
                score = 0
                count = 0
                for word1 in topic:
                    for word2 in topic:
                        if word1 == word2: continue
                        if word1 not in glove or word2 not in glove: continue
                        score += glove.similarity(word1, word2)
                        count += 1

                results['c_we_per_topic'].append(0 if count == 0 else score / count)
            results['c_we'] = np.mean(results['c_we_per_topic'])
            results['c_we_std'] = np.std(results['c_we_per_topic'])

            return results

        while True:
            try:
                self.log.debug('creating coherence model')
                coherence_model = gensim.models.coherencemodel.CoherenceModel(topics=topic_words, texts=text,
                                                                              dictionary=dictionary,
                                                                              coherence=metric)
                coherence_per_topic = coherence_model.get_coherence_per_topic()

                topic_coherence = [coherence_per_topic[i] for i, t in enumerate(self.topics)]

                results[metric + '_per_topic'] = topic_coherence
                results[metric] = np.nanmean(coherence_per_topic)
                results[metric + '_std'] = np.nanstd(coherence_per_topic)

                break

            except KeyError as e:
                key = str(e)[1:-1]
                for x in topic_words:
                    if key in x:
                        x.remove(key)

        return results

    def evaluate(self, labels_pred: list, labels_true: list, metric='purity', average_method='arithmetic'):
        """Evaluation against a ground truth

        :param list labels_pred: Predicted topics
        :param list labels_true: Ground truth labels
        :param str metric: Metric for computing the evaluation, among <purity, homogeneity, completeness, v-measure, nmi>
        :param str average_method: Only if metric is NMI, the average method among <arithmetic, min, max, geometric>
        """

        unique_labels = list(np.unique(labels_true))
        l = [unique_labels.index(x) for x in labels_true]
        if type(labels_pred[0]) == list:
            p = [x[0][0] if len(x) > 0 else 0 for x in labels_pred]
        else:
            p = labels_pred

        if metric == 'purity':
            cm = contingency_matrix(l, p)
            return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        elif metric == 'homogeneity':
            return metrics.homogeneity_score(l, p)
        elif metric == 'completeness':
            return metrics.completeness_score(l, p)
        elif metric == 'v-measure':
            return metrics.v_measure_score(l, p)
        elif metric == 'nmi':
            return metrics.normalized_mutual_info_score(l, p, average_method)
        else:
            raise ValueError(f'Unrecognised metrics: {metric}')
