import warnings

from .abstract_model import AbstractModel
from .doc2topic import models, corpora

ROOT = ''

MODEL_PATH = ROOT + '/models/ntm/ntm'


# Neural Topic Model
class NtmModel(AbstractModel):
    # Load the saved model
    def load(self):
        self.model = models.Doc2Topic()
        self.model.load(filename=MODEL_PATH)

    def get_raw_topics(self):
        json_topics = {}
        topic_words = []

        topics = self.model.get_topic_words()

        for i, topic in topics.items():
            json_topics[str(i)] = {'words': {}}
            topic_words.append([])
            for word, weight in topic:
                json_topics[str(i)]['words'][word] = float(weight)
                topic_words[-1].append(word)

        return json_topics, topic_words

    # Get topic-word distribution
    def topics(self):
        if self.model is None:
            self.load()
        topics = self.model.get_topic_words()

        json_topics = {}

        for i, topic in topics.items():
            json_topics[str(i)] = {}
            json_topics[str(i)]['words'] = {}
            for word, weight in topic:
                json_topics[str(i)]['words'][word] = float(weight)

        return json_topics

    def get_corpus_predictions(self):
        if self.model is None:
            self.load()

        return [self.model.get_document_topics(i) for i in range(0, len(self.model.get_docvecs()))]

    # Train the model
    def train(self,
              datapath='/data/data.txt',
              n_topics=35,
              batch_size=1024 * 6,
              n_epochs=20,
              lr=0.05,
              l1_doc=0.000002,
              l1_word=0.000000015,
              word_dim=None,
              generator=None):

        warnings.filterwarnings("ignore")

        models.init_tf_memory()
        data = corpora.DocData(datapath)

        self.model = models.Doc2Topic()

        self.model.build(data, n_topics=n_topics, batch_size=batch_size, n_epochs=n_epochs, lr=lr, l1_doc=l1_doc,
                         l1_word=l1_word, word_dim=word_dim, generator=generator)

        fmeasure = self.model.history.history['fmeasure'][-1]
        loss = self.model.history.history['loss'][-1]

        self.model.save(MODEL_PATH)
        return 'success', fmeasure, loss

    def predict(self, doc, topn=5):
        return {'message': 'not implemented for this model'}
