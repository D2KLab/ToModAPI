from gensim import corpora
from gensim.models import HdpModel

from .abstract_model import AbstractModel
from .gensim_model import GensimModel
from .utils.corpus import preprocess, input_to_list_string


class HDPModel(GensimModel):
    """Hierarchical Dirichlet Processing Model

    Source: https://radimrehurek.com/gensim/models/hdpmodel.html
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/hdp'):
        super().__init__(model_path)

    def train(self, data=AbstractModel.ROOT + '/data/test.txt',
              preprocessing=False,
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
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param int max_chunks: Upper bound on how many chunks to process. It wraps around corpus beginning in another corpus pass, if there are not enough chunks in the corpus.
            :param int max_time: Upper bound on time (in seconds) for which model will be trained.
            :param int chunksize: Number of documents in one chuck.
            :param float kappa: Learning parameter which acts as exponential decay factor to influence extent of learning from each batch.
            :param float tau: Learning parameter which down-weights early iterations of documents.
            :param int max_time: Upper bound on time (in seconds) for which model will be trained.
            :param float K: Second level truncation level
            :param float T: Top level truncation level
            :param float alpha: Second level concentration
            :param float gamma: First level concentration
            :param float eta: The topic Dirichlet
            :param float scale: Weights information from the mini-chunk of corpus to calculate rhot.
            :param float var_converge: Lower bound on the right side of convergence. Used when updating variational parameters for a single document.
            :param str outputdir: Stores topic and options information in the specified directory.
            :param str random_state: Adds a little random jitter to randomize results around same alpha.
        """
        data = input_to_list_string(data, preprocessing)

        if preprocessing:
            data = map(preprocess, data)

        texts = [
            [token for token in text.split(' ') if len(token) > 0]
            for text in data
        ]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        self.model = HdpModel(corpus, id2word=dictionary,
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

        self.dictionary = dictionary
        self.corpus_predictions = self.model[corpus]

        return 'success'
