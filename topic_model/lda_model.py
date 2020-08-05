from os import path
import pickle

from .abstract_model import AbstractModel
import gensim

MALLET_PATH = path.join(path.dirname(__file__), 'mallet-2.0.8', 'bin', 'mallet')


# Latent Dirichlet Allocation
class LdaModel(AbstractModel):
    def __init__(self, model_path=AbstractModel.ROOT + '/models/lda/lda.pkl',
                 mallet_dep_path=AbstractModel.ROOT + '/models/mallet-dep/'):
        super().__init__()
        self.model_path = model_path
        self.mallet_dep_path = mallet_dep_path

    # Load saved model
    def load(self):
        with open(self.model_path, "rb") as input_file:
            self.model = pickle.load(input_file)
        self.model.mallet_path = MALLET_PATH
        self.model.prefix = self.mallet_dep_path

    # Perform Inference
    def predict(self, doc, topn=5):
        if self.model is None:
            self.load()

        # Transform document into BoW
        doc = doc.split()
        common_dictionary = self.model.id2word
        doc = common_dictionary.doc2bow(doc)
        # Get topic distribution
        doc_topic_dist = self.model[doc]
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

    def train(self,
              datapath=AbstractModel.ROOT + '/data/data.txt',
              num_topics=35,
              alpha=50,
              random_seed=5,
              iter=500,
              optimize_interval=10,
              topic_threshold=0.0):
        """Train LDA model.

            :param datapath: The path of the training corpus
            :param int num_topics: The desired number of topics
            :param float alpha: Prior document-topic distribution
            :param int random_seed: Random seed to ensure consistent results, if 0 - use system clock
            :param int iter: Number of iteration in EM
            :param int optimize_interval: Hyperparameter optimization every optimize_interval
            :param float topic_threshold:  Threshold of the probability above which we consider a topic
        """

        # Load data
        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        # Transform documents
        tokens = [doc.split() for doc in text]

        id2word = gensim.corpora.Dictionary(tokens)
        id2word.filter_n_most_frequent(20)

        corpus = [id2word.doc2bow(doc) for doc in tokens]

        self.log.debug('start training LDA')
        # Train the model
        self.model = gensim.models.wrappers.LdaMallet(MALLET_PATH,
                                                      corpus=corpus,
                                                      num_topics=num_topics,
                                                      alpha=alpha,
                                                      id2word=id2word,
                                                      random_seed=random_seed,
                                                      prefix=self.mallet_dep_path,
                                                      iterations=iter,
                                                      optimize_interval=optimize_interval,
                                                      topic_threshold=topic_threshold)

        # Save the model
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

        self.log.debug('end training LDA')

        return 'success'

    def topics(self):
        topics = []

        for i in range(0, self.model.num_topics):
            words = []
            weights = []
            for word, weight in self.model.show_topic(i, topn=10):
                weights.append(weight)
                words.append(word)

            topics.append({
                'words': words,
                'weights': weights
            })

        return topics
