import os
import warnings

from .abstract_model import AbstractModel


# Neural Topic Model
class Doc2TopicModel(AbstractModel):
    def __init__(self, model_path=AbstractModel.ROOT + '/models/doc2topic', name='d2t'):
        global models, corpora
        super().__init__()
        from .doc2topic import models, corpora

        self.model_path = model_path
        self.name = name
        os.makedirs(model_path, exist_ok=True)

    # Load the saved model
    def load(self):
        self.model = models.Doc2Topic()
        self.model.load(filename=os.path.join(self.model_path, self.name))

    def topics(self):
        if self.model is None:
            self.load()
        topics = []

        for i, topic in self.model.get_topic_words().items():
            words = []
            weights = []
            for word, weight in topic:
                words.append(word)
                weights.append(float(weight))
            topics.append({'words': words,
                           'weights': weights})

        return topics

    def get_corpus_predictions(self):
        if self.model is None:
            self.load()

        return [self.model.get_document_topics(i) for i in range(0, len(self.model.get_docvecs()))]

    # Train the model
    def train(self,
              datapath=AbstractModel.ROOT + '/data/data.txt',
              n_topics=35,
              batch_size=1024 * 6,
              n_epochs=20,
              lr=0.05,
              l1_doc=0.000002,
              l1_word=0.000000015,
              word_dim=None,
              generator=None, return_scores=False):

        warnings.filterwarnings("ignore")

        models.init_tf_memory()
        data = corpora.DocData(datapath)

        self.model = models.Doc2Topic()

        self.model.build(data, n_topics=n_topics, batch_size=batch_size, n_epochs=n_epochs, lr=lr, l1_doc=l1_doc,
                         l1_word=l1_word, word_dim=word_dim, generator=generator)

        fmeasure = self.model.history.history['fmeasure'][-1]
        loss = self.model.history.history['loss'][-1]

        self.model.save(os.path.join(self.model_path, self.name))
        self.log.debug(f'Training complete. F-measure: {fmeasure}. Loss: {loss}')
        if return_scores:
            return 'success', fmeasure, loss
        else:
            return 'success'

    def predict(self, doc, topn=5):
        return {'message': 'not implemented for this model'}
