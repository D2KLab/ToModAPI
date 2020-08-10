import pickle
import gensim

from collections import defaultdict
from utils.corpus import preprocess

from gensim import models, corpora
from gensim.models import LsiModel

from abstract_model import AbstractModel


# Latent Semantic Indexing Model
class LSIModel(AbstractModel):
    """Latent Semantic Indexing Model

    Source: https://radimrehurek.com/gensim/models/lsimodel.html
    """

    def __init__(self, name='LSI'):
        """LFTM Model constructor
        :param name: Name of the model
        """
        super().__init__()

        self.model = None
        self.corpus = None
        self.dictionairy = None


    def train(self, data, num_topics: int, 
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
            tfidf = models.TfidfModel(corpus)
            corpus = tfidf[corpus]

        lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics,
                                    chunksize=chunksize, decay=decay, distributed=distributed, onepass=distributed, 
                                    power_iters=power_iters, extra_samples=power_iters)

        self.model = lsi_model
        self.dictionary = dictionary
        self.corpus = lsi_model[corpus]
        
        return 'success'


    # Perform Inference
    def predict(self, text, topn=10, preprocessing=True):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
        """
        if preprocessing:
            text = preprocess(text)
        preds = self.model[self.dictionary.doc2bow(text.split())]
        
        return sorted(preds, key=lambda x: -abs(x[1]))[:topn]


    def get_corpus_predictions(self, topn: int = 5):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
        """
        topics = [sorted(doc, key=lambda x: -abs(x[1]))[:topn] for doc in self.corpus]
        return topics

    @property
    def topics(self):
        n_top_words = 10
        topics = []
        for topic_words in self.model.get_topics():
            topics.append([(x[1][1], x[0]) for x in sorted(zip(topic_words, self.dictionary.items()), key=lambda x: -x[0])[:n_top_words]])

        return topics
