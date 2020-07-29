import re
import os
from os import path
import pickle
import subprocess
from .utils.LoggerWrapper import LoggerWrapper
from .abstract_model import AbstractModel
import gensim

LFTM_JAR = path.join(path.dirname(__file__), 'lftm', 'LFTM.jar')
GLOVE_TOKENS = path.join(path.dirname(__file__), 'glove', 'glovetokens.pkl')
GLOVE_TXT = path.join(path.dirname(__file__), 'glove', 'glove.6B.50d.txt')

TOPIC_REGEX = r'Topic(\d+): (.+)'


# Function to remove specified tokens from a string
def remove_tokens(x, tok2remove):
    return ' '.join(['' if t in tok2remove else t for t in x.split()])


# Latent Feature Topic Model
class LftmModel(AbstractModel):
    def __init__(self, model_root=AbstractModel.ROOT + '/models/lftm', data_root=AbstractModel.ROOT + '/data/lftm',
                 name='LFLDA'):
        """
        LFTM Model constructor

        Parameters:
        model_root (path): Path of the computed model
        data_root (path): Path of output files, regenerated at each prediction
        name (str): Name of the model
        """
        super().__init__()

        model_root = path.abspath(model_root)
        self.top_words = model_root + '/%s.topWords' % name
        self.paras_path = model_root + '/%s.paras' % name
        self.theta_path_model = model_root + '/%s.theta' % name
        self.data_glove = model_root + '/%s.glove' % name

        data_root = path.abspath(data_root)
        self.doc_path = data_root + '/doc.txt'
        self.theta_path = data_root + '/%sinf.theta' % name

        self.name = name
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(model_root, exist_ok=True)

    # Perform Inference
    def predict(self, doc, topn=10, initer=500, niter=0):
        """
            doc: the document on which to make the inference
            topn: number of the most probable topical words
            initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
            niter: sampling iterations for the latent feature topic models
        """
        with open(GLOVE_TOKENS, "rb") as input_file:
            glovetokens = pickle.load(input_file)

        params = {}
        with open(self.paras_path, "r") as f:
            for line in f.readlines():
                k, v = line.strip().split('\t')
                params[k[1:]] = v

        doc = ' '.join([word for word in doc.split() if word in glovetokens])

        with open(self.doc_path, "w", encoding='utf-8') as f:
            f.write(doc)

        # Perform Inference
        proc = f'java -jar {LFTM_JAR} -model {params["model"]}inf -paras {self.paras_path} -corpus {self.doc_path} ' \
               f'-initers {initer} -niters {niter} -twords {topn} -name {self.name}inf -sstep 0'
        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)
        completed_proc = subprocess.run(proc, shell=True, stderr=logWrap, stdout=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        with open(self.theta_path, "r") as file:
            doc_topic_dist = file.readline()

        doc_topic_dist = [(topic, float(weight)) for topic, weight in enumerate(doc_topic_dist.split())]
        sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]
        results = [{topic: weight} for topic, weight in sorted_doc_topic_dist]
        return results

    # Train the model
    def train(self,
              datapath=AbstractModel.ROOT + '/data/data.txt',
              ntopics=35,
              alpha=0.1,
              beta=0.1,
              _lambda=1,
              initer=50,
              niter=5,
              topn=10,
              model='LFLDA'):
        """
            datapath: the path to the training text file
            model: topic model, LFLDA (default) or LFDMM.
            ntopics: the number of topics
            alpha: prior document-topic distribution
            beta: prior topic-word distribution
            lambda: mixture weight
            initer: initial sampling iterations to separate the counts for the latent feature component
                    and the Dirichlet multinomial component
            niter: sampling iterations for the latent feature topic models
            topn: number of the most probable topical words
        """
        if model not in ['LFLDA', 'LFDMM']:
            raise ValueError('Model should be LFLDA (default) or LFDMM.')

        with open(GLOVE_TOKENS, "rb") as input_file:
            glovetokens = pickle.load(input_file)

        with open(datapath, "r", encoding='utf-8') as datafile:
            text = [line.rstrip() for line in datafile if line]

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

        proc = f'java -jar {LFTM_JAR} -model LFLDA -corpus {self.data_glove} -vectors {GLOVE_TXT} -ntopics {ntopics} ' \
               f'-alpha {alpha} -beta {beta} -lambda {_lambda} -initers {initer} -niters {niter} -twords {topn} ' \
               f'-name {self.name} -sstep 0'
        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)

        completed_proc = subprocess.run(proc, shell=True, stdout=logWrap, stderr=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        return 'success' if completed_proc.returncode == 0 else ('error %d' % completed_proc.returncode)

    def topics(self):
        topics = []

        with open(self.top_words, 'r') as f:
            for line in f:
                l = line.strip()
                match = re.match(TOPIC_REGEX, l)
                if not match:
                    continue
                _id, words = match.groups()
                topics.append({'words': words.split()})

        return topics

    def get_corpus_predictions(self):
        with open(self.theta_path_model, "r") as file:
            doc_topic_dist = [line.strip().split() for line in file.readlines()]

        topics = [[(i, float(score)) for i, score in enumerate(doc)]
                  for doc in doc_topic_dist]

        topics = [sorted(doc, key=lambda t: -t[1]) for doc in topics]
        return topics
