import numpy as np
import gensim
import logging


class AbstractModel:
    ROOT = '.'

    def __init__(self):
        self.model = None
        self.log = logging.getLogger(self.__class__.__name__)

    def load(self):
        """
            Load the model and eventual dependencies.
            Implementation not mandatory.
        """
        pass

    # Perform Inference
    def predict(self, text, topn=5, preprocessing=False):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocess: If True, execute preprocessing on the document
        """
        raise NotImplementedError

    def predict_corpus(self, datapath=ROOT + '/data/data.txt', topn=5):
        if self.model is None:
            self.load()

        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        return [self.predict(t, topn=topn) for t in text]

    def train(self, data=ROOT + '/data.txt', num_topics=35, preprocessing=False):
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

    def coherence(self, datapath=ROOT + '/data/data.txt', coherence='c_v'):
        """ Get the coherence of the topic mode.

        :param datapath: Path of the corpus on which compute the coherence.
        :param coherence: Type of coherence to compute, among <c_v, c_npmi, c_uci, u_mass>
         """
        topics = self.topics
        topic_words = [x['words'] for x in topics]

        self.log.debug('loading dataset')
        with open(datapath, "r") as datafile:
            text = [line.rstrip().split() for line in datafile if line]

        dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

        results = {}

        while True:
            try:
                self.log.debug('creating coherence model')
                coherence_model = gensim.models.coherencemodel.CoherenceModel(topics=topic_words, texts=text,
                                                                              dictionary=dictionary,
                                                                              coherence=coherence)
                coherence_per_topic = coherence_model.get_coherence_per_topic()

                for i, t in enumerate(self.topics):
                    t[coherence] = coherence_per_topic[i]

                results['topics'] = topics
                results[coherence] = np.nanmean(coherence_per_topic)
                results[coherence + '_std'] = np.nanstd(coherence_per_topic)

                break

            except KeyError as e:
                key = str(e)[1:-1]
                for x in topic_words:
                    if key in x:
                        x.remove(key)

        return results
