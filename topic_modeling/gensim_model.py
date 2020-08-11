import os
import pickle
from collections import defaultdict

from gensim import corpora
from gensim.models import LsiModel, TfidfModel

from .abstract_model import AbstractModel
from .utils.corpus import preprocess, input_to_list_string


class GensimModel(AbstractModel):
    """Skeleton for models imported from Gensim"""

    def __init__(self, model_path=None):
        super().__init__(model_path)

        self.corpus_predictions = None
        self.dictionary = None

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'rb') as f:
            self.dictionary = pickle.load(f)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'rb') as f:
            self.corpus_predictions = pickle.load(f)

    def save(self, path=None):
        super().save(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'wb') as f:
            pickle.dump(self.corpus_predictions, f, pickle.HIGHEST_PROTOCOL)

    def predict(self, text, topn=10, preprocessing=True):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
        """
        if self.model is None:
            self.load()

        if preprocessing:
            text = preprocess(text)
        preds = self.model[self.dictionary.doc2bow(text.split())]

        return sorted(preds, key=lambda x: -abs(x[1]))[:topn]

    def get_corpus_predictions(self, topn: int = 5):
        if self.corpus_predictions is None:
            self.load()

        topics = [sorted(doc, key=lambda x: -abs(x[1]))[:topn] for doc in self.corpus_predictions]
        return topics

    @property
    def topics(self):
        if self.model is None:
            self.load()

        n_top_words = 10
        topics = []
        for topic_weight in self.model.get_topics():
            sorted_words = sorted(zip(topic_weight, self.dictionary.items()), key=lambda x: -x[0])[:n_top_words]

            topics.append({
                'words': [t[1][1] for t in sorted_words],
                'weights': [t[0] for t in sorted_words]
            })

        return topics

    def topic(self, topic_id: int):
        if self.model is None:
            self.load()

        sorted_words = sorted(self.model.show_topic(topic_id, topn=10), key=lambda x: -x[1])
        return {
            'words': [t[1] for t in sorted_words],
            'weights': [t[0] for t in sorted_words]
        }
