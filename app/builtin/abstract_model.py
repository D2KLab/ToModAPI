import numpy as np
import gensim


class AbstractModel:
    def __init__(self):
        self.model = None

    def load(self):
        """
            Load the model and eventual dependencies.
            Implementation not mandatory.
        """
        pass

    # Perform Inference
    def predict(self, doc, topn=5):
        """
            doc: text on which to perform inference
            topn: the number of top keywords to extract
        """
        raise NotImplementedError

    def predict_corpus(self, datapath='/data/data.txt'):
        if self.model is None:
            self.load()

        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        return [self.predict(t) for t in text]

    def train(self, datapath='/data/data.txt'):
        """
            datapath: path to training data text file
        """
        raise NotImplementedError

    def get_raw_topics(self):
        """
            Returns
            - json_topics
            - topic_words
        """
        raise NotImplementedError

    def topics(self):
        raise NotImplementedError

    def get_corpus_predictions(self):
        """
        Returns the predictions computed on the training corpus.
        This is not re-computing predictions, but reading training results.
        """
        raise NotImplementedError

    def coherence(self, datapath='/data/data.txt', coherence='c_v'):
        """ Get coherence of model topics """
        if self.model is None:
            self.load()
        json_topics, topic_words = self.get_raw_topics()

        print('loading dataset')
        with open(datapath, "r") as datafile:
            text = [line.rstrip().split() for line in datafile if line]

        dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

        while True:
            try:
                print('creating coherence model')
                coherence_model = gensim.models.coherencemodel.CoherenceModel(topics=topic_words, texts=text,
                                                                              dictionary=dictionary,
                                                                              coherence=coherence)
                coherence_per_topic = coherence_model.get_coherence_per_topic()

                for i in range(len(topic_words)):
                    json_topics[str(i)][coherence] = coherence_per_topic[i]

                json_topics[coherence] = np.nanmean(coherence_per_topic)
                json_topics[coherence + '_std'] = np.nanstd(coherence_per_topic)

                break

            except KeyError as e:
                key = str(e)[1:-1]
                for x in topic_words:
                    if key in x:
                        x.remove(key)

        return json_topics
