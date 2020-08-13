import os
import pickle
import gensim
import shutil

from .utils.corpus import preprocess, input_to_list_string
from .abstract_model import AbstractModel

MALLET_PATH = os.path.join(os.path.dirname(__file__), 'mallet-2.0.8', 'bin', 'mallet')


class LdaModel(AbstractModel):
    """Latent Dirichlet Allocation

    Source: https://radimrehurek.com/gensim/models/ldamodel.html
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/lda/'):
        super().__init__(model_path)
        mallet_dep_path = os.path.join(self.model_path, 'mallet-dep/')
        os.makedirs(mallet_dep_path, exist_ok=True)

    def train(self,
              data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              alpha=50,
              random_seed=5,
              iter=500,
              optimize_interval=10,
              topic_threshold=0.0):
        """Train LDA model.

            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param float alpha: Prior document-topic distribution
            :param int random_seed: Random seed to ensure consistent results, if 0 - use system clock
            :param int iter: Number of iteration in EM
            :param int optimize_interval: Hyperparameter optimization every optimize_interval
            :param float topic_threshold:  Threshold of the probability above which we consider a topic
        """
        text = input_to_list_string(data, preprocessing)

        # Transform documents
        tokens = [doc.split() for doc in text]

        id2word = gensim.corpora.Dictionary(tokens)
        id2word.filter_n_most_frequent(20)

        corpus = [id2word.doc2bow(doc) for doc in tokens]

        mallet_dep_path = os.path.join(self.model_path, 'mallet-dep/')

        self.log.debug('start training LDA')
        # Train the model
        self.model = gensim.models.wrappers.LdaMallet(MALLET_PATH,
                                                      corpus=corpus,
                                                      num_topics=num_topics,
                                                      alpha=alpha,
                                                      id2word=id2word,
                                                      random_seed=random_seed,
                                                      prefix=mallet_dep_path,
                                                      iterations=iter,
                                                      optimize_interval=optimize_interval,
                                                      topic_threshold=topic_threshold)

        self.log.debug('end training LDA')

        return 'success'

    def save(self, path=None):
        if path is not None and path != self.model_path:
            shutil.move(os.path.join(self.model_path, 'mallet-dep/'), os.path.join(path, 'mallet-dep/'))

        super().save(path)

        with open(os.path.join(self.model_path, 'lda.pkl'), 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'lda.pkl'), "rb") as input_file:
            self.model = pickle.load(input_file)
        self.model.mallet_path = MALLET_PATH
        self.model.prefix = os.path.join(self.model_path, 'mallet-dep/')

    def predict(self, text, topn=5, preprocessing=False):
        if self.model is None:
            self.load()

        if preprocessing:
            text = preprocess(text)

        # Transform document into BoW
        text = text.split()
        common_dictionary = self.model.id2word
        text = common_dictionary.doc2bow(text)
        # Get topic distribution
        doc_topic_dist = self.model[text]
        # Sort to get the top n topics
        # Structure the results into a dictionary
        results = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]

        return results

    def get_corpus_predictions(self, topn: int = 5):
        if self.model is None:
            self.load()

        topics = self.model.load_document_topics()
        topics = [sorted(doc, key=lambda t: -t[1])[:topn] for doc in topics]

        return topics

    @property
    def topics(self):
        if self.model is None:
            self.load()

        return [self.topic(i) for i in range(0, self.model.num_topics)]

    def topic(self, topic_id: int):
        if self.model is None:
            self.load()

        words = []
        weights = []
        for word, weight in self.model.show_topic(topic_id, topn=10):
            weights.append(weight)
            words.append(word)
        return {
            'words': words,
            'weights': weights
        }
