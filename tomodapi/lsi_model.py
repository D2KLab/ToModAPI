from collections import defaultdict

from gensim import corpora
from gensim.models import LsiModel, TfidfModel

from .abstract_model import AbstractModel
from .gensim_model import GensimModel
from .utils.corpus import preprocess, input_to_list_string


class LSIModel(GensimModel):
    """Latent Semantic Indexing

    Source: https://radimrehurek.com/gensim/models/lsimodel.html
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/lsi'):
        super().__init__(model_path)

    def train(self, data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
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
