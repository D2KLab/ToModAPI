import os
import pickle
import gensim

from .abstract_model import AbstractModel
from .gsdmm import MovieGroupProcess
from .utils.corpus import preprocess, input_to_list_string


class GsdmmModel(AbstractModel):
    """Gibbs Sampling Algorithm for a Dirichlet Mixture Model

    Source: https://github.com/rwalk/gsdmm
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/gsdmm/'):
        super().__init__(model_path)

    def train(self,
              data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              alpha=0.1,
              beta=0.1,
              iter=15):
        """Train GSDMM model.

            :param data: The path of the training corpus
            :param int num_topics: The desired number of topics (upper bound)
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param float alpha: Prior document-topic distribution
            :param float beta: Prior topic-word distribution
            :param int iter: Sampling iterations for the latent feature topic models
        """
        self.model = MovieGroupProcess(K=num_topics, alpha=alpha, beta=beta, n_iters=iter)

        text = input_to_list_string(data, preprocessing)
        tokens = [doc.split() for doc in text]
        id2word = gensim.corpora.Dictionary(tokens)

        self.log.debug('start training GSDMM')
        self.model.fit(tokens, len(id2word), log=self.log.debug)
        self.log.debug('end training GSDMM')

        return 'success'

    def save(self, path=None):
        super().save(path)

        with open(os.path.join(self.model_path, 'gsdmm.pkl'), 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'gsdmm.pkl'), "rb") as input_file:
            self.model = pickle.load(input_file)

    @property
    def topics(self):
        if self.model is None:
            self.load()

        topics = []
        for i, topic in enumerate(self.model.cluster_word_distribution):
            current_words = []
            current_freq = []
            total = sum(topic.values())
            for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse=True)[:10]:
                current_words.append(word)
                current_freq.append(freq / total)

            topics.append({
                'words': current_words,
                'weights': current_freq
            })

        return topics

    def predict(self, text: str, topn=5, preprocessing=False, doc_len=7):
        if self.model is None:
            self.load()

        if preprocessing:
            preprocess(text)

        # gsdmm works for short text
        # given the preprocessing, here there is no punctuation nor stopwords
        # we keep the first N words
        text = text.split()[0:doc_len]

        results = [(topic, score) for topic, score in enumerate(self.model.score(text))]
        results = sorted(results, key=lambda kv: kv[1], reverse=True)[:topn]
        return results

    def get_corpus_predictions(self, topn=5):
        if self.model is None:
            self.load()

        topics = [[(topic, score) for topic, score in enumerate(doc)] for doc in self.model.doc_cluster_scores]
        topics = [sorted(doc, key=lambda t: -t[1])[:topn] for doc in topics]
        return topics
