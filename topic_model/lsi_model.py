import os
import pickle
from collections import defaultdict

from gensim import corpora
from gensim.models import LsiModel, TfidfModel

from .abstract_model import AbstractModel
from .utils.corpus import preprocess, input_to_list_string


# Latent Semantic Indexing Model
class LSIModel(AbstractModel):
    """Latent Semantic Indexing

    Source: https://radimrehurek.com/gensim/models/lsimodel.html
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/lsi'):
        """LSI Model constructor
        """
        super().__init__()

        self.model_path = model_path
        self.model = None
        self.corpus_predictions = None
        self.dictionary = None

    def train(self, data=AbstractModel.ROOT + '/test.txt', num_topics=35,
              preprocessing=True,
              use_tfidf=True,
              chunksize=20000,
              decay=1.0,
              distributed=False,
              onepass=True,
              power_iters=2,
              extra_samples=100):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param bool use_tfidf: If true, use TF-iDF instead of word frequency
            :param int chunksize: Number of documents to be used in each training chunk.
            :param int decay: Weight of existing observations relatively to new ones
            :param bool distributed:  If True - distributed mode (parallel execution on several machines) will be used.
            :param bool onepass: Whether the one-pass algorithm should be used for training. Pass False to force a multi-pass stochastic algorithm.
            :param int power_iters: Number of power iteration steps to be used. Increasing the number of power iterations improves accuracy, but lowers performance
            :param int extra_samples:  Extra samples to be used besides the rank k. Can improve accuracy.
        """
        frequency = defaultdict(int)
        data = input_to_list_string(data, preprocessing)
        for text in data:
            for token in text.split(' '):
                frequency[token] += 1

        if preprocessing:
            data = map(preprocess, data)

        texts = [
            [token for token in text.split(' ') if frequency[token] > 1 and len(token) > 0]
            for text in data
        ]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        if use_tfidf:
            tfidf = TfidfModel(corpus)
            corpus = tfidf[corpus]

        lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=num_topics,
                             chunksize=chunksize, decay=decay, distributed=distributed, onepass=distributed,
                             power_iters=power_iters, extra_samples=power_iters)

        self.model = lsi_model
        self.dictionary = dictionary
        self.corpus_predictions = lsi_model[corpus]

        return 'success'

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'rb') as f:
            self.dictionary = pickle.load(f)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'rb') as f:
            self.corpus_predictions = pickle.load(f)

    def save(self, path=None):
        super().save(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'wb') as f:
            pickle.dump(self.corpus_predictions, f, pickle.HIGHEST_PROTOCOL)

    def predict(self, text, topn=10, preprocessing=True):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
        """
        if self.model is None:
            self.load()

        if preprocessing:
            text = preprocess(text)
        preds = self.model[self.dictionary.doc2bow(text.split())]

        return sorted(preds, key=lambda x: -abs(x[1]))[:topn]

    def get_corpus_predictions(self, topn: int = 5):
        if self.corpus_predictions is None:
            self.load()

        topics = [sorted(doc, key=lambda x: -abs(x[1]))[:topn] for doc in self.corpus_predictions]
        return topics

    @property
    def topics(self):
        if self.model is None:
            self.load()

        n_top_words = 10
        topics = []
        for topic_weight in self.model.get_topics():
            sorted_words = sorted(zip(topic_weight, self.dictionary.items()), key=lambda x: -x[0])[:n_top_words]

            topics.append({
                'words': [t[1][1] for t in sorted_words],
                'weights': [t[0] for t in sorted_words]
            })

        return topics

    def topic(self, topic_id: int):
        if self.model is None:
            self.load()

        sorted_words = sorted(self.model.show_topic(topic_id, topn=10), key=lambda x: -x[1])
        return {
            'words': [t[1] for t in sorted_words],
            'weights': [t[0] for t in sorted_words]
        }
