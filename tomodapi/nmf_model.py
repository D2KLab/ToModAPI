from collections import defaultdict

from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf

from .gensim_model import GensimModel
from .abstract_model import AbstractModel
from .utils.corpus import preprocess, input_to_list_string


class NMFModel(GensimModel):
    """Non-Negative Matrix factorization

    Source: https://radimrehurek.com/gensim/models/nmf.html
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/nmf'):
        super().__init__(model_path)

    def train(self, data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              passes=1,
              kappa=1.0,
              minimum_probability=0.01,
              w_max_iter=200,
              w_stop_condition=0.0001,
              h_max_iter=50,
              h_stop_condition=0.001,
              eval_every=10,
              normalize=True,
              random_state=None):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param int passes: Number of full passes over the training corpus. Leave at default passes=1 if your input is an iterator.
            :param float kappa: Gradient descent step size. Larger value makes the model train faster, but could lead to non-convergence if set too large.
            :param float minimum_probability: If normalize is True, topics with smaller probabilities are filtered out. If normalize is False, topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s.
            :param float w_max_iter: Maximum number of iterations to train W per each batch.
            :param float w_stop_condition: If error difference gets less than that, training of W stops for the current batch.
            :param float h_max_iter: Maximum number of iterations to train h per each batch.
            :param float h_stop_condition: If error difference gets less than that, training of h stops for the current batch.
            :param int eval_every: Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if set too low.
            :param bool normalize:  Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.
            :param int random_state: Seed for random generator. Needed for reproducibility.

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

        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        nmf_model = Nmf(corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        kappa=kappa,
                        minimum_probability=minimum_probability,
                        w_max_iter=w_max_iter,
                        w_stop_condition=w_stop_condition,
                        h_max_iter=h_max_iter,
                        h_stop_condition=h_stop_condition,
                        eval_every=eval_every,
                        normalize=normalize,
                        random_state=random_state)

        self.model = nmf_model
        self.dictionary = dictionary
        self.corpus_predictions = nmf_model[corpus]

        return 'success'
