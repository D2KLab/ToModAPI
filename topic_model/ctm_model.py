import pickle
import scipy

from utils.corpus import preprocess
import numpy as np

from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list

from abstract_model import AbstractModel


# Contextualized Topic Model
class CTMModel(AbstractModel):
    """Contextualized Topic Model

    Source: https://github.com/MilaNLProc/contextualized-topic-models
    """

    def __init__(self, name='CTM', sbert_model_to_use='distiluse-base-multilingual-cased',):
        """LFTM Model constructor
        :param name: Name of the model
        """
        super().__init__()

        self.model = None
        self.corpus = None
        self.dictionairy = None
        self.bow = None

        self.sbert_model_to_use = sbert_model_to_use


    def train(self, data, 
              num_topics: int,
              preprocessing=True, 
              bert_input_size=512, 
              num_epochs=10, 
              hidden_sizes = (100, ),
              inference_type="contextual", 
              num_data_loader_workers=0):
        """
        Train the model and generate the results on the corpus
            :param data: The training corpus as path or list of strings
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param str sbert_model_to_use='distiluse-base-multilingual-cased',
            :param int bert_input_size: size of bert embeddings, 
            :param int num_epochs: number of epochs for trainng the model, 
            :param tuple hidden_sizes: n_layers,
            :param str inference_type: "contextual" or "combined", 
            :param int num_data_loader_workers: number of data loader workers (default cpu_count). set it to 0 if you are using Windows
        """
        
        if preprocessing:
            data = list(map(preprocess, data))

        indptr  = [0]
        indices = []
        ones    = []
        vocabulary = {}

        for d in data:
            for term in d.split():
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                ones.append(1)
            indptr.append(len(indices))
 
        idx2token = {v: k for (k, v) in vocabulary.items()}
        bow = scipy.sparse.csr_matrix((ones, indices, indptr), dtype=int)

        bert_embeddings = bert_embeddings_from_list(data, self.sbert_model_to_use)
        ctm_model = CTM(input_size=len(vocabulary), bert_input_size=bert_input_size, num_epochs=num_epochs, inference_type=inference_type, n_components=num_topics)

        training_dataset = CTMDataset(bow, bert_embeddings, idx2token)
        ctm_model.fit(training_dataset)

        self.bow    = bow
        self.model  = ctm_model
        self.corpus = ctm_model.get_thetas(training_dataset)
        self.vocabulary = vocabulary

        return 'success'


    # Perform Inference
    def predict(self, text, topn=10, preprocessing=True, n_trials=10):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
        """
        if preprocessing:
            text = preprocess(text)

        indices_test = []
        data_test = []
        indptr_test = [0]

        for term in text.split():
            if term not in self.vocabulary: continue
            index = self.vocabulary[term]
            indices_test.append(index)
            data_test.append(1)

        indices_test.append(len(self.vocabulary) - 1)
        data_test.append(0)
        indptr_test.append(len(indices_test))

        bow_test = scipy.sparse.csr_matrix((data_test, indices_test, indptr_test), dtype=int)
        bert_embeddings = bert_embeddings_from_list([text], self.sbert_model_to_use)
        testing_dataset = CTMDataset(bow_test, bert_embeddings, [])

        thetas = np.zeros((len(testing_dataset), len(self.model.get_topic_lists())))

        for a in range(0, n_trials):
            thetas = thetas + self.model.get_thetas(testing_dataset)
            
        thetas = thetas[0]/np.sum(thetas[0])
        preds = zip(range(len(thetas)), thetas)
        
        return sorted(preds, key=lambda x: -x[1])[:topn]


    def get_corpus_predictions(self, topn: int = 5):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
        """
        topics = [sorted(zip(range(len(x)), x), key=lambda x: -x[1])[:topn] for x in self.corpus]
        return topics

    @property
    def topics(self):
        """
            returns a list of words associated with each topic
        """
        n_top_words = 10
        topics = self.model.get_topic_lists(n_top_words)

        return topics
