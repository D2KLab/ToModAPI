import joblib
import numpy as np
from pvtm import pvtm

from .abstract_model import AbstractModel
from .utils.corpus import preprocess, input_to_list_string


class PvtmModel(AbstractModel):
    """Paragraph Vector Topic Model

    Source: https://github.com/davidlenz/pvtm
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/pvtm/pvtm.gz'):
        super().__init__()
        self.model_path = model_path

    def train(self,
              data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=35,
              preprocessing=False,
              vector_size=50,
              hs=0,
              dbow_words=1,
              dm=0,
              epochs=30,
              window=1,
              seed=123,
              min_count=0,
              workers=1,
              alpha=0.025,
              min_alpha=0.025,
              random_state=123,
              covariance_type='diag'):
        """Train LDA model.

            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param int vector_size: dimensionality of the feature vectors (Doc2Vec)
            :param int hs: If 1, hierarchical softmax will be used for model training. If set to 0, negative sampling will be used. (Doc2Vec) Value among <0, 1>
            :param int dbow_words:	If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors  (Doc2Vec)  Value among <0, 1>
            :param int dm:	Distributed bag of words (word2vec-Skip-Gram) (dm=0) OR distributed memory (dm=1).  Value among <0, 1>
            :param int epochs:	training epochs (Doc2Vec)
            :param int window: window size (Doc2Vec)
            :param seed: seed for the random number generator (Doc2Vec)
            :param min_count: minimal number of appearences for a word to be considered (Doc2Vec)
            :param int workers:	number workers (Doc2Vec)
            :param float alpha:	initial learning rate (Doc2Vec)
            :param float min_alpha:	doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
            :param int random_state: random seed (GMM)
            :param str covariance_type:	covariance type (GMM) among <diag, spherical, tied, full>
        """
        text = input_to_list_string(data, preprocessing)

        self.model = pvtm.PVTM(text)
        self.model.fit(vector_size=vector_size, n_components=num_topics, hs=hs, dbow_words=dbow_words, dm=dm,
                       epochs=epochs, window=window, seed=seed, min_count=min_count, workers=workers, alpha=alpha,
                       min_alpha=min_alpha, random_state=random_state, covariance_type=covariance_type)
        self.log.debug('end training PVTM')

        return 'success'

    def save(self, path=None):
        super().save()
        self.model.save(path=self.model_path)

    def load(self, path=None):
        super().load()
        self.model = joblib.load(self.model_path)

    def predict(self, text, topn=5, preprocessing=False, ):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocess: If True, execute preprocessing on the document
        """

        if self.model is None:
            self.load()

        if preprocessing:
            text = preprocess(text)

        # print(self.model.get_document_topics())
        results = [(topic, score) for topic, score in enumerate(self.model.infer_topics(text, probabilities=True)[0])]
        results = sorted(results, key=lambda kv: kv[1], reverse=True)[:topn]

        return results

    @property
    def topics(self):
        if self.model is None:
            self.load()

        return [self.topic(x) for x in np.arange(len(self.model.wordcloud_df))]

    def topic(self, topic_id: int):
        if self.model is None:
            self.load()

        all_topic_words = self.model.wordcloud_df.loc[topic_id].split()
        words, counts = np.unique(all_topic_words, return_counts=True)
        count_sort_ind = np.argsort(-counts)[:10]

        return {
            'words': words[count_sort_ind].tolist(),
            'weights': np.divide(counts[count_sort_ind], len(all_topic_words)).tolist()
        }

    def get_corpus_predictions(self, topn: int = 5):
        if self.model is None:
            self.load()

        topics = [self.model.get_topic_weights(doc.reshape(1, -1), probabilities=True)
                  for doc in self.model.doc_vectors]
        topics = [[(i, x) for i, x in enumerate(doc[0])] for doc in topics]
        topics = [sorted(doc, key=lambda t: -t[1])[:topn] for doc in topics]

        return topics
