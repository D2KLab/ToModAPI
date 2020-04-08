from os import path
import pickle

from .abstract_model import AbstractModel
import gensim

MALLET_PATH = path.join(path.dirname(__file__), 'mallet-2.0.8', 'bin', 'mallet')

ROOT = ''
MODEL_PATH = ROOT + '/models/lda/lda.pkl'
MALLET_DEP = ROOT + '/models/mallet-dep/'
W2V_PATH = ROOT + '/data/word2vec.bin'


# Latent Dirichlet Allocation
class LdaModel(AbstractModel):
    # Load saved model
    def load(self):
        with open(MODEL_PATH, "rb") as input_file:
            self.model = pickle.load(input_file)
        self.model.mallet_path = MALLET_PATH
        self.model.prefix = MALLET_DEP

    # Perform Inference
    def predict(self, doc, topn=5):
        if self.model is None:
            self.load()

        # Transform document into BoW
        doc = doc.split()
        common_dictionary = self.model.id2word
        doc = common_dictionary.doc2bow(doc)
        # Get topic distribution
        doc_topic_dist = self.model[doc]
        # Sort to get the top n topics
        # Structure the results into a dictionary
        results = [{topic: weight} for topic, weight in
                   sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]]

        return results

    # Train the model
    def train(self,
              datapath=ROOT + '/data/data.txt',
              num_topics=35,
              alpha=50,
              random_seed=5,
              iterations=500,
              optimize_interval=10,
              topic_threshold=0.0):
        """
            datapath: path to training data text file
            num_topics: number of topics
            alpha: prior document-topic distribution
            iternations: number of iteration in EM
            optimize_interval: hyperparameter optimization every optimize_interval
            topic_threshold:  threshold of the probability above which we consider a topic
        """

        # Load data
        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        # Transform documents
        tokens = [doc.split() for doc in text]

        id2word = gensim.corpora.Dictionary(tokens)
        id2word.filter_n_most_frequent(20)

        corpus = [id2word.doc2bow(doc) for doc in tokens]

        print('start training LDA')
        # Train the model
        self.model = gensim.models.wrappers.LdaMallet(MALLET_PATH,
                                                      corpus=corpus,
                                                      num_topics=num_topics,
                                                      alpha=alpha,
                                                      id2word=id2word,
                                                      random_seed=random_seed,
                                                      prefix=MALLET_DEP,
                                                      iterations=iterations,
                                                      optimize_interval=optimize_interval,
                                                      topic_threshold=topic_threshold)

        # Save the model
        with open(MODEL_PATH, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

        print('end training LDA')

        return 'success'

    def get_raw_topics(self):
        json_topics = {}
        topic_words = []

        for i in range(0, self.model.num_topics):
            json_topics[str(i)] = {
                'words': {}
            }
            topic_words.append([])
            for word, weight in self.model.show_topic(i, topn=10):
                json_topics[str(i)]['words'][word] = weight
                topic_words[-1].append(word)
        return json_topics, topic_words

    # Get topic-word distribution
    def topics(self):
        if self.model is None:
            self.load()

        json_topics = {}
        for i in range(0, self.model.num_topics):
            json_topics[str(i)] = {
                'words': [word for word, weight in self.model.show_topic(i, topn=10)]
            }

        return json_topics

    # Get weighted similarity of topic words and tags
    def evaluate(self, datapath=ROOT + '/data/data.txt', tagspath=ROOT + '/data/tags.txt', topn=5):
        # Load a KeyedVector model using a pre-trained word2vec
        word2vecmodel = gensim.models.KeyedVectors.load(W2V_PATH, mmap='r')
        # Load vocabulary
        vocab = word2vecmodel.wv.vocab

        # Extract and transform text
        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        score = 0
        num_doc = 0

        common_dictionary = self.model.id2word
        tokens = [doc.split() for doc in text]
        corpus = [common_dictionary.doc2bow(doc) for doc in tokens]

        all_doc_topic_dist = self.model[corpus]

        file2 = open(tagspath)

        # Iterate over each document
        while True:

            tags = file2.readline()

            if not tags:
                break

            # Retrieve top n topics
            doc_topic_dist = all_doc_topic_dist[num_doc]
            sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]

            print('doc', num_doc)
            doc_score = 0
            topic_weights = 0
            # Iterate over each topic-word distribution
            for topic_id, topic_weight in sorted_doc_topic_dist:
                topic_words = self.model.show_topic(topic_id, topn=topn)
                max_similarity = []
                word_id = 0
                word_weights = 0
                # Iterate over the words in the topic
                for word, weight in topic_words:
                    # Check if word has a vector
                    if word in vocab:
                        max_similarity.append(0)
                        # Get max similarity with tags
                        for tag in tags.split(' '):
                            if tag in vocab and 'ted' not in tag.lower():
                                similarity = weight * word2vecmodel.similarity(word, tag)
                                if similarity > max_similarity[word_id]:
                                    max_similarity[word_id] = similarity

                        word_id += 1
                        word_weights += weight
                # Get topic score
                topic_score = sum(max_similarity) / word_weights
                # Update document score
                doc_score += topic_weight * topic_score
                topic_weights += topic_weight
            # Get document score
            doc_score /= topic_weights
            print('doc score', doc_score)

            score += doc_score
            num_doc += 1
        # Get total score for all documents
        score /= num_doc

        # Return score
        return score

    def get_corpus_predictions(self):
        if self.model is None:
            self.load()

        return list(self.model.load_document_topics())
