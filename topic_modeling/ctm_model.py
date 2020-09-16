import os
import pickle
import scipy

import numpy as np
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list

from .utils.corpus import preprocess, input_to_list_string
from .abstract_model import AbstractModel

SBERT_MODEL = 'distiluse-base-multilingual-cased'


class CTMModel(AbstractModel):
    """Contextualized Topic Model

    Source: https://github.com/MilaNLProc/contextualized-topic-models
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/ctm'):
        super().__init__(model_path)

        self.corpus_predictions = None
        self.bow = None
        self.dictionary = None

    def train(self, data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              bert_input_size=512,
              num_epochs=100,
              hidden_sizes=(100,),
              batch_size=200,
              inference_type="contextual",
              num_data_loader_workers=0):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param int bert_input_size: Size of bert embeddings
            :param int num_epochs: Number of epochs for trainng the model,
            :param tuple hidden_sizes: n_layers,
            :param int batch_size: Batch size
            :param str inference_type: Inference type among <contextual, combined>
            :param int num_data_loader_workers: Number of data loader workers (default cpu_count). Set it to 0 if you are using Windows
        """
        data = input_to_list_string(data, preprocessing)

        if preprocessing:
            data = list(map(preprocess, data))

        indptr = [0]
        indices = []
        ones = []
        vocabulary = {}

        for d in data:
            for term in d.split():
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                ones.append(1)
            indptr.append(len(indices))

        idx2token = {v: k for (k, v) in vocabulary.items()}
        bow = scipy.sparse.csr_matrix((ones, indices, indptr), dtype=int)

        bert_embeddings = bert_embeddings_from_list(data, SBERT_MODEL)
        ctm_model = CTM(input_size=len(vocabulary), bert_input_size=bert_input_size, num_epochs=num_epochs,
                        inference_type=inference_type, n_components=num_topics, batch_size=batch_size)

        training_dataset = CTMDataset(bow, bert_embeddings, idx2token)
        ctm_model.fit(training_dataset)

        self.bow = bow
        self.model = ctm_model
        self.corpus_predictions = ctm_model.get_thetas(training_dataset)
        self.dictionary = vocabulary

        return 'success'

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'rb') as f:
            self.dictionary = pickle.load(f)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'rb') as f:
            self.corpus_predictions = pickle.load(f)

        with open(os.path.join(self.model_path, 'bow.pkl'), 'rb') as f:
            self.bow = pickle.load(f)

    def save(self, path=None):
        super().save(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'wb') as f:
            pickle.dump(self.corpus_predictions, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'bow.pkl'), 'wb') as f:
            pickle.dump(self.bow, f, pickle.HIGHEST_PROTOCOL)

    def predict(self, text, topn=10, preprocessing=True, n_trials=10):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
            :param int n_trials: Number of inference to compute and average
        """
        if self.model is None:
            self.load()

        if preprocessing:
            text = preprocess(text)

        indices_test = []
        data_test = []
        indptr_test = [0]

        for term in text.split():
            if term not in self.dictionary:
                continue
            index = self.dictionary[term]
            indices_test.append(index)
            data_test.append(1)

        indices_test.append(len(self.dictionary) - 1)
        data_test.append(0)
        indptr_test.append(len(indices_test))

        bow_test = scipy.sparse.csr_matrix((data_test, indices_test, indptr_test), dtype=int)
        bert_embeddings = bert_embeddings_from_list([text], SBERT_MODEL)
        testing_dataset = CTMDataset(bow_test, bert_embeddings, [])

        thetas = np.zeros((len(testing_dataset), len(self.model.get_topic_lists())))

        for a in range(0, n_trials):
            thetas = thetas + self.model.get_thetas(testing_dataset)

        thetas = thetas[0] / np.sum(thetas[0])
        preds = zip(range(len(thetas)), thetas)

        return sorted(preds, key=lambda x: -x[1])[:topn]

    def get_corpus_predictions(self, topn: int = 5):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
        """
        if self.model is None:
            self.load()

        topics = [sorted(zip(range(len(x)), x), key=lambda x: -x[1])[:topn] for x in self.corpus_predictions]
        return topics

    @property
    def topics(self):
        """
            Returns a list of words associated with each topic
        """
        if self.model is None:
            self.load()

        n_top_words = 10
        topics = []

        for word_list in self.model.get_topic_lists(n_top_words):
            topics.append({
                'words': word_list
            })

        return topics
