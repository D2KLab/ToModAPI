import re
import pickle
import subprocess

from .abstract_model import AbstractModel
import gensim

ROOT = '/app'
LFTM_JAR = ROOT + '/modules/lftm/jar/LFTM.jar'

DATA_ROOT = ROOT + '/data'
TOP_WORDS = DATA_ROOT + '/lftm/TEDLFLDA.topWords'
THETA_PATH = DATA_ROOT + '/lftm/TEDLFLDAinf.theta'
PARAS_PATH = DATA_ROOT + '/lftm/TEDLFLDA.paras'
DOC_PATH = DATA_ROOT + '/lftm/doc.txt'
GLOVE_TOKENS = DATA_ROOT + '/glovetokens.pkl'
DATA_GLOVE = DATA_ROOT + '/lftm/data_glove.txt'
GLOVE_TXT = DATA_ROOT + '/glove.6B.50d.txt'
W2V_BIN = DATA_ROOT + '/word2vec.bin'

TOPIC_REGEX = r'Topic(\d+): (.+)'


# Function to remove specified tokens from a string
def remove_tokens(x, tok2remove):
    return ' '.join(['' if t in tok2remove else t for t in x.split()])


# Latent Feature Topic Model
class LftmModel(AbstractModel):

    # Perform Inference
    def predict(self,
                doc,
                initer=500,
                niter=0,
                topn=10,
                name='TEDLFLDAinf'):
        """
            doc: the document on which to make the inference
            initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
            niter: sampling iterations for the latent feature topic models
            topn: number of the most probable topical words
            name: prefix of the inference documents
        """
        with open(GLOVE_TOKENS, "rb") as input_file:
            glovetokens = pickle.load(input_file)

        doc = ' '.join([word for word in doc.split() if word in glovetokens])

        with open(DOC_PATH, "w", encoding='utf-8') as f:
            f.write(doc)

        # Perform Inference
        completedProc = subprocess.run(
            'java -jar {} -model {} -paras {} -corpus {} -initers {} -niters {} -twords '
            '{} -name {} -sstep {}'.format(
                LFTM_JAR,
                'LFLDAinf',
                PARAS_PATH,
                DOC_PATH,
                str(initer),
                str(niter),
                str(topn),
                name,
                '0'), shell=True)

        # os.system('mv /app/data/TEDLFLDAinf.* /app/models/lftm/')

        print(completedProc.returncode)

        file = open(THETA_PATH, "r")
        doc_topic_dist = file.readline()
        file.close()

        doc_topic_dist = [(topic, float(weight)) for topic, weight in enumerate(doc_topic_dist.split())]
        sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:5]
        results = [{topic: weight} for topic, weight in sorted_doc_topic_dist]
        return results

    # Train the model
    def train(self,
              datapath=DATA_ROOT + '/data.txt',
              ntopics=35,
              alpha=0.1,
              beta=0.1,
              _lambda=1,
              initer=50,
              niter=5,
              topn=10):
        """
            datapath: the path to the training text file
            ntopics: the number of topics
            alpha: prior document-topic distribution
            beta: prior topic-word distribution
            lambda: mixture weight
            initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
            niter: sampling iterations for the latent feature topic models
            topn: number of the most probable topical words
        """

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

        file = open(DATA_GLOVE, "w")
        for doc in text:
            file.write(doc + '\n')
        file.close()

        completedProc = subprocess.run(
            'java -jar {} -model {} -corpus {} -vectors {} -ntopics {} -alpha {} -beta {} -lambda {} -initers {} -niters {} -twords {} -name {} -sstep {}'.format(
                LFTM_JAR,
                'LFLDA',
                DATA_GLOVE,
                GLOVE_TXT,
                str(ntopics),
                str(alpha),
                str(beta),
                str(_lambda),
                str(initer),
                str(niter),
                str(topn),
                'TEDLFLDA',
                '0'), shell=True)

        print(completedProc.returncode)

        return 'success'

    def get_raw_topics(self):
        json_topics = {}
        topic_words = []

        with open(TOP_WORDS, 'r') as f:
            for line in f:
                l = line.strip()
                match = re.match(TOPIC_REGEX, l)
                if not match:
                    continue
                _id, words = match.groups()
                topics = words.split()
                json_topics[_id] = {'words': topics}
                topic_words.append(topics)

        return json_topics, topic_words

    # Get topic-word distribution
    def topics(self):
        json_topics, topic_words = self.get_raw_topics()
        return json_topics

    # Get weighted similarity of topic words and tags
    def evaluate(self, tagspath=DATA_ROOT + '/tags.txt', topn=5):

        # Load a KeyedVector model using a pre-trained word2vec
        word2vec = gensim.models.KeyedVectors.load(W2V_BIN, mmap='r')
        # Load vocabulary
        vocab = word2vec.wv.vocab

        # Prepare Topics
        _, topics = self.get_raw_topics()
        topics = [t[:topn] for t in topics]

        # Prepare document-topic distribution
        doc_topic_dist = []
        with open(THETA_PATH) as f:
            for line in f:
                txt = line.strip()
                if not txt:
                    continue
                doc_topic_dist.append([float(doc) for doc in txt.split()][:topn])

        with open(tagspath) as f:
            score = 0
            # Iterate over each document
            for num_doc, line in enumerate(f):
                tags = line.strip()
                if not tags:
                    continue

                print('doc', num_doc)
                doc_score = 0
                topic_weights = 0
                # Iterate over the top topics
                for topic_id, topic_weight in enumerate(doc_topic_dist[num_doc]):
                    topic_words = topics[int(topic_id)]
                    max_similarity = []
                    word_id = 0
                    # Iterate over topic words
                    for word in topic_words:
                        if word in vocab:
                            max_similarity.append(0)
                            # Compute the maximum similarity with tags
                            for tag in tags.split(' '):
                                if tag in vocab and 'ted' not in tag.lower():
                                    similarity = word2vec.similarity(word, tag)
                                    if similarity > max_similarity[word_id]:
                                        max_similarity[word_id] = similarity

                            word_id += 1
                    topic_score = sum(max_similarity) / word_id
                    doc_score += topic_weight * topic_score
                    topic_weights += topic_weight
                doc_score /= topic_weights
                print('doc score', doc_score)

                score += doc_score

        score /= (num_doc + 1)

        print(score)
        return score
