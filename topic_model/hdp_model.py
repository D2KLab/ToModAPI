import pickle
import gensim

from collections import defaultdict

from gensim import models, corpora
from gensim.models import HdpModel
from .utils.corpus import preprocess

from .abstract_model import AbstractModel


class HDPModel(AbstractModel):
    """Hierarchical Dirichlet Processing Model

    Source: https://radimrehurek.com/gensim/models/hdpmodel.html
    """

    def __init__(self):
        super().__init__()

        self.model = None
        self.corpus = None
        self.dictionary = None

    def train(self, data,
              preprocessing=True,
              max_chunks=None,
              max_time=None,
              chunksize=256,
              kappa=1.0,
              tau=64.0,
              K=15,
              T=150,
              alpha=1,
              gamma=1,
              eta=0.01,
              scale=1.0,
              var_converge=0.0001,
              outputdir=None,
              random_state=None):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param int chunksize: Upper bound on how many chunks to process. It wraps around corpus beginning in another corpus pass, if there are not enough chunks in the corpus.
            :param float kappa: Learning parameter which acts as exponential decay factor to influence extent of learning from each batch.
            :param float tau: Learning parameter which down-weights early iterations of documents.
            :param int max_time: Upper bound on time (in seconds) for which model will be trained.
            :param float K: Second level truncation level
            :param float T: Top level truncation level
            :param float alpha: Second level concentration
            :param float gamma: First level concentration
            :param float eta: The topic Dirichlet
            :param float eta: Weights information from the mini-chunk of corpus to calculate rhot.
            :param float var_converge: Lower bound on the right side of convergence. Used when updating variational parameters for a single document.
            :param str outputdir: Stores topic and options information in the specified directory.
            :param str outputdir: Stores topic and options information in the specified directory.
        """

        if preprocessing:
            data = map(preprocess, data)

        texts = [
            [token for token in text.split(' ') if len(token) > 0]
            for text in data
        ]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lsi_model = models.HdpModel(corpus, id2word=dictionary,
                                    max_chunks=max_chunks,
                                    max_time=max_time,
                                    chunksize=chunksize,
                                    kappa=kappa,
                                    tau=tau,
                                    K=K,
                                    T=T,
                                    alpha=alpha,
                                    gamma=gamma,
                                    eta=eta,
                                    scale=scale,
                                    var_converge=var_converge,
                                    outputdir=outputdir,
                                    random_state=random_state)

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
        """
        topics = [sorted(doc, key=lambda x: -abs(x[1]))[:topn] for doc in self.corpus]
        return topics

    @property
    def topics(self):
        n_top_words = 10
        topics = []
        for topic_words in self.model.get_topics():
            topics.append([(x[1][1], x[0]) for x in
                           sorted(zip(topic_words, self.dictionary.items()), key=lambda x: -x[0])[:n_top_words]])

        return topics
