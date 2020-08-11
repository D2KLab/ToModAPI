import re
import os
import pickle
import subprocess
import gensim
import shutil

from .utils.LoggerWrapper import LoggerWrapper
from .abstract_model import AbstractModel
from .utils.corpus import preprocess, input_to_list_string

LFTM_JAR = os.path.join(os.path.dirname(__file__), 'lftm', 'LFTM.jar')
GLOVE_TOKENS = os.path.join(os.path.dirname(__file__), 'glove', 'glovetokens.pkl')
GLOVE_TXT = os.path.join(os.path.dirname(__file__), 'glove', 'glove.6B.50d.txt')

TOPIC_REGEX = r'Topic(\d+): (.+)'


# Function to remove specified tokens from a string
def remove_tokens(x, tok2remove):
    return ' '.join(['' if t in tok2remove else t for t in x.split()])


# Latent Feature Topic Model
class LftmModel(AbstractModel):
    """Latent Feature Topic Model

    Source: https://github.com/datquocnguyen/LFTM
    """

    def __init__(self, model_path=AbstractModel.ROOT + '/models/lftm', data_root=AbstractModel.ROOT + '/data/lftm',
                 name='LFLDA'):
        """LFTM Model constructor

        :param model_path: Path of the computed model
        :param data_root: Path of output files, regenerated at each prediction
        :param name: Name of the model
        """
        super().__init__(model_path)

        self.top_words = None
        self.paras_path = None
        self.theta_path_model = None
        self.data_glove = None

        self.update_model_path(model_path, name)

        data_root = os.path.abspath(data_root)
        self.doc_path = data_root + '/doc.txt'
        self.theta_path = data_root + '/%sinf.theta' % name

        self.name = name
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    def update_model_path(self, model_root, name):
        model_root = os.path.abspath(model_root)
        self.model_path = model_root
        self.top_words = model_root + '/%s.topWords' % name
        self.paras_path = model_root + '/%s.paras' % name
        self.theta_path_model = model_root + '/%s.theta' % name
        self.data_glove = model_root + '/%s.glove' % name

    def save(self, path=None):
        if path is not None and path != self.model_path:
            shutil.move(self.model_path, path)
            self.model_path = path

    def train(self,
              data=AbstractModel.ROOT + '/data/test.txt',
              num_topics=20,
              preprocessing=False,
              alpha=0.1,
              beta=0.1,
              _lambda=1,
              initer=50,
              niter=5,
              twords=10,
              model='LFLDA'):
        """ Train LFTM model.

            :param data: The path of the training corpus
            :param int num_topics: The desired number of topics
            :param bool preprocessing: If true, apply preprocessing to the corpus
            :param float alpha: Prior document-topic distribution
            :param float beta: Prior topic-word distribution
            :param int _lambda: Mixture weight
            :param int initer: Initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
            :param int niter: Sampling iterations for the latent feature topic models
            :param int twords: Number of topical words to return
            :param model: Topic model, among <LFLDA, LFDMM>.
        """
        if model not in ['LFLDA', 'LFDMM']:
            raise ValueError('Model should be LFLDA (default) or LFDMM.')

        with open(GLOVE_TOKENS, "rb") as input_file:
            glovetokens = pickle.load(input_file)

        text = input_to_list_string(data, preprocessing)
        tokens = [doc.split() for doc in text]

        id2word = list(gensim.corpora.Dictionary(tokens).values())

        tok2remove = {}
        for t in id2word:
            if t not in glovetokens:
                tok2remove[t] = True

        text = [remove_tokens(doc, tok2remove) for doc in text]

        with open(self.data_glove, "w") as file:
            for doc in text:
                file.write(doc + '\n')

        proc = f'java -jar {LFTM_JAR} -model LFLDA -corpus {self.data_glove} -vectors {GLOVE_TXT} -ntopics {num_topics} ' \
               f'-alpha {alpha} -beta {beta} -lambda {_lambda} -initers {initer} -niters {niter} -twords {twords} ' \
               f'-name {self.name} -sstep 0'
        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)

        completed_proc = subprocess.run(proc, shell=True, stdout=logWrap, stderr=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        return 'success' if completed_proc.returncode == 0 else ('error %d' % completed_proc.returncode)

    # Perform Inference
    def predict(self, text, topn=10, preprocessing=False, initer=500, niter=0):
        """Predict topic of the given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocessing: If True, execute preprocessing on the document
            :param int initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
            :param int niter: sampling iterations for the latent feature topic models
        """
        with open(GLOVE_TOKENS, "rb") as input_file:
            glovetokens = pickle.load(input_file)

        if preprocessing:
            preprocess(text)

        params = {}
        with open(self.paras_path, "r") as f:
            for line in f.readlines():
                k, v = line.strip().split('\t')
                params[k[1:]] = v

        text = ' '.join([word for word in text.split() if word in glovetokens])

        with open(self.doc_path, "w", encoding='utf-8') as f:
            f.write(text)

        # Perform Inference
        proc = f'java -jar {LFTM_JAR} -model {params["model"]}inf -paras {self.paras_path} -corpus {self.doc_path} ' \
               f'-initers {initer} -niters {niter} -twords {topn} -name {self.name}inf -sstep 0'
        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)
        completed_proc = subprocess.run(proc, shell=True, stderr=logWrap, stdout=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        with open(self.theta_path, "r") as file:
            doc_topic_dist = file.readline()

        doc_topic_dist = [(int(topic), float(weight)) for topic, weight in enumerate(doc_topic_dist.split())]
        results = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]
        return results

    def get_corpus_predictions(self, topn: int = 5):
        with open(self.theta_path_model, "r") as file:
            doc_topic_dist = [line.strip().split() for line in file.readlines()]

        topics = [[(i, float(score)) for i, score in enumerate(doc)]
                  for doc in doc_topic_dist]

        topics = [sorted(doc, key=lambda t: -t[1])[:topn] for doc in topics]
        return topics

    @property
    def topics(self):
        topics = []
        with open(self.top_words, 'r') as f:
            for line in f:
                match = re.match(TOPIC_REGEX, line.strip())
                if not match:
                    continue
                _id, words = match.groups()
                topics.append({'words': words.split()})

        return topics
