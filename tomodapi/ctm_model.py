import os
import pickle
import scipy

import numpy as np
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list

from .utils.corpus import preprocess, input_to_list_string
from .abstract_model import AbstractModel


class CTMModel(AbstractModel):
    """Contextualized Topic Model

    Source: https://github.com/MilaNLProc/contextualized-topic-models
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/ctm'):
        super().__init__(model_path)

        self.bert_model = None
        self.corpus_predictions = None
        self.dictionary = None

    def train(self, data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              bert_model='distilbert-base-nli-mean-tokens',
              contextual_size=768,
              num_epochs=100,
              hidden_sizes=(100,),
              batch_size=200,
              inference_type="combined",
              model_type='prodLDA',
              activation='softplus',
              dropout=0.2,
              learn_priors=True,
              lr=2e-3,
              momentum=0.99,
              solver='adam',
              reduce_on_plateau=False,
              num_data_loader_workers=0):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param string bert_model: BERT pretraining to use (see https://www.sbert.net/docs/pretrained_models.html)
            :param int contextual_size: Size of bert embeddings
            :param int num_epochs: Number of epochs for training the model,
            :param tuple hidden_sizes: n_layers,
            :param int batch_size: Batch size
            :param str inference_type: Inference type among <contextual, combined>
            :param str model_type: Model type among <prodLDA, LDA>
            :param str activation: Activation among <softplus, relu>
            :param float dropout: dropout to use (default 0.2)
            :param bool learn_priors: If true, make priors a learnable parameter (default True)
            :param float lr: Learning rate to use for training (default 2e-3)
            :param float momentum: Momentum to use for training (default 0.99)
            :param str solver: Optimizer among <adam, sgd>
            :param bool reduce_on_plateau: If true, reduce learning rate by 10x on plateau of 10 epochs (default False)
            :param int num_data_loader_workers: Number of data loader workers (default cpu_count). Set it to 0 if you are using Windows
        """
        self.bert_model = bert_model
        data = input_to_list_string(data, preprocessing)

        if preprocessing:
            data_prep = list(map(preprocess, data))
        else:
            data_prep = data

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

        qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")
        training_dataset = qt.fit(text_for_contextual=data, text_for_bow=data_prep)

        md = ZeroShotTM if inference_type == 'zeroshot' else CombinedTM
        ctm_model = md(bow_size=len(vocabulary), contextual_size=contextual_size, num_epochs=num_epochs,
                       model_type=model_type, hidden_sizes=hidden_sizes, activation=activation,
                       dropout=dropout, learn_priors=learn_priors, lr=lr, momentum=momentum,
                       solver=solver, reduce_on_plateau=reduce_on_plateau, n_components=num_topics,
                       batch_size=batch_size)

        ctm_model.fit(training_dataset)

        self.model = ctm_model
        self.qt = qt
        self.corpus_predictions = ctm_model.get_thetas(training_dataset)
        self.dictionary = vocabulary

        return 'success'

    def load(self, path=None):
        super().load(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'rb') as f:
            self.dictionary = pickle.load(f)

        with open(os.path.join(self.model_path, 'qt.pkl'), 'rb') as f:
            self.qt = pickle.load(f)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'rb') as f:
            self.corpus_predictions = pickle.load(f)

        with open(os.path.join(self.model_path, 'model.txt'), 'r') as f:
            self.bert_model = f.read(self.bert_model)

    def save(self, path=None):
        super().save(path)

        with open(os.path.join(self.model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'qt.pkl'), 'wb') as f:
            pickle.dump(self.qt, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'corpus_predictions.pkl'), 'wb') as f:
            pickle.dump(self.corpus_predictions, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_path, 'model.txt'), 'w') as f:
            f.write(self.bert_model)

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
            text_prep = preprocess(text)
        else:
            text_prep = text

        if isinstance(self.model, CombinedTM):
            testing_dataset = self.qt.transform(text_for_contextual=[text], text_for_bow=[text_prep])
        else:
            testing_dataset = self.qt.transform(text_for_contextual=[text])

        preds = self.model.get_doc_topic_distribution(testing_dataset, n_samples=20)[0]
        preds = [(i, p) for i, p in enumerate(preds)]
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
