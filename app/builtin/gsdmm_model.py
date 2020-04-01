import pickle
import gensim

from .abstract_model import AbstractModel
from .gsdmm import MovieGroupProcess

MODEL_PATH = '/app/models/gsdmm/gsdmm.pkl'


# Gibbs Sampling Algorithm for a Dirichlet Mixture Model
class GsdmmModel(AbstractModel):
    # Load the saved model
    def load(self):
        with open(MODEL_PATH, "rb") as input_file:
            self.model = pickle.load(input_file)

    def get_raw_topics(self):
        json_topics = {}
        topic_words = []

        for i, topic in enumerate(self.model.cluster_word_distribution):
            json_topics[str(i)] = {
                'words': {}
            }
            topic_words.append([])
            total = sum(topic.values())
            for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse=True)[:10]:
                json_topics[str(i)]['words'][word] = freq / total
                topic_words[-1].append(word)

        return json_topics, topic_words

    # Get topic-word distribution
    def topics(self):
        if self.model is None:
            self.load()

        json_topics = {}

        for i, topic in enumerate(self.model.cluster_word_distribution):
            json_topics[str(i)] = {
                'words': []
            }
            for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse=True)[:10]:
                json_topics[str(i)]['words'].append(word)

        return json_topics

    # Train the model
    def train(self,
              datapath='/app/data/data.txt',
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

        # Fit the model
        self.model.fit(tokens, len(id2word))

        # Save the new model
        with open(MODEL_PATH, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

        return 'success'

    # Perform Inference
    def predict(self, doc, topn=5):
        if self.model is None:
            self.load()
        results = [(topic, score) for topic, score in enumerate(self.model.score(doc))]
        print(results)
        results = [{topic: weight} for topic, weight in sorted(results, key=lambda kv: kv[1], reverse=True)[:topn]]
        return results
