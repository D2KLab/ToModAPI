import pickle
import gensim

from .abstract_model import AbstractModel
from .gsdmm import MovieGroupProcess


# Gibbs Sampling Algorithm for a Dirichlet Mixture Model
class GsdmmModel(AbstractModel):
    def __init__(self, model_path=AbstractModel.ROOT + '/models/gsdmm/gsdmm.pkl'):
        super().__init__()
        self.model_path = model_path

    # Load the saved model
    def load(self):
        with open(self.model_path, "rb") as input_file:
            self.model = pickle.load(input_file)

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

    # Train the model
    def train(self,
              datapath=AbstractModel.ROOT + '/data/data.txt',
              n_topics=35,
              alpha=0.1,
              beta=0.1,
              n_iter=15):

        # Build the model
        self.model = MovieGroupProcess(K=n_topics, alpha=alpha, beta=beta, n_iters=n_iter)

        # Load data
        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        # Transform documents
        tokens = [doc.split() for doc in text]

        id2word = gensim.corpora.Dictionary(tokens)

        self.log.debug('start training GSDMM')
        # Fit the model
        self.model.fit(tokens, len(id2word), log=self.log.debug)
        self.log.debug('end training GSDMM')

        # Save the new model
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

        return 'success'

    # Perform Inference
    def predict(self, doc, topn=5, doc_len=7):
        if self.model is None:
            self.load()

        # gsdmm works for short text
        # given the preprocessing, here there is no punctuation nor stopwords
        # we keep the first N words
        doc = doc.split()[0:doc_len]

        results = [(topic, score) for topic, score in enumerate(self.model.score(doc))]
        results = [{topic: weight} for topic, weight in sorted(results, key=lambda kv: kv[1], reverse=True)[:topn]]
        return results

    def get_corpus_predictions(self):
        if self.model is None:
            self.load()

        topics = [[(topic, score) for topic, score in enumerate(doc)] for doc in self.model.doc_cluster_scores]
        topics = [sorted(doc, key=lambda t: -t[1]) for doc in topics]
        return topics
