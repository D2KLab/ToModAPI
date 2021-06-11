import csv
import heapq
import json
import random
from os.path import isfile

import numpy as np
from tensorflow.keras.layers import Input, Embedding, dot, Reshape, Activation, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.metrics.pairwise import cosine_similarity

from .measures import fmeasure

L2 = (lambda x: np.linalg.norm(x, 2))
L1 = (lambda x: np.linalg.norm(x, 1))
L1normalize = (lambda x: np.divide(x, L1(x), out=np.zeros(len(x)), where=L1(x) != 0))
cosine = (lambda a, b: np.dot(a, b) / (L2(a) * L2(b)) if sum(a) != 0 and sum(b) != 0 else 0)
relufy = np.vectorize(lambda x: max(0., x))


def init_tf_memory():
    # Config GPU memory usage
    from tensorflow.compat import v1 as tf
    tf.disable_v2_behavior()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = -1
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


class Doc2Topic:
    """ doc2topic model class """

    def __init__(self):
        self.topic_words = None
        self.wordvecs = None
        self.docvecs = None
        self.layer_lookup = None

    def build(self, corpus, n_topics=20, batch_size=1024 * 6, n_epochs=5, lr=0.015, l1_doc=0.000002,
              l1_word=0.000000015, word_dim=None, generator=None):
        init_tf_memory()
        self.corpus = corpus

        self.idx2token = self.corpus.idx2token
        self.token2idx = self.corpus.token2idx

        self.params = {'Ntopics': n_topics,
                       'Ndocs': self.corpus.n_docs,
                       'BS': batch_size,
                       'LR': lr,
                       'L1doc': l1_doc,
                       'L1word': l1_word,
                       'NS': self.corpus.ns_rate}
        self.topic_words = None
        self.wordvecs = None
        self.docvecs = None
        self.generator = generator

        inlayerD = Input((1,))
        embD = Embedding(self.corpus.n_docs, n_topics, input_length=1, trainable=True, activity_regularizer=l1(l1_doc),
                         name="docvecs")(inlayerD)
        embDa = Activation('relu')(embD)
        embD = Reshape((n_topics, 1))(embDa)

        inlayerW = Input((1,))
        if word_dim:  # Experimental setting: extra dense layer for projecting word vectors onto document vector space
            embW = Embedding(self.corpus.vocab_size, word_dim, input_length=1, trainable=True,
                             activity_regularizer=l1(l1_word), name="wordemb")(inlayerW)
            embWa = Dense(n_topics, activation='relu', activity_regularizer=l1(l1_word), name="wordproj")(embW)
            embW = Reshape((n_topics, 1))(embWa)
        else:
            embW = Embedding(self.corpus.vocab_size, n_topics, input_length=1, trainable=True,
                             activity_regularizer=l1(l1_word), name="wordvecs")(inlayerW)
            embWa = Activation('relu')(embW)
            embW = Reshape((n_topics, 1))(embWa)

        # sim = dot([embD, embW], 0, normalize=True)
        dot_prod = dot([embD, embW], 1, normalize=False)
        dot_prod = Reshape((1,))(dot_prod)

        output = Activation('sigmoid')(dot_prod)

        opt = Adam(lr=lr, amsgrad=True)

        self.model = Model(inputs=[inlayerD, inlayerW], outputs=[output])
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[fmeasure])
        self.layer_lookup = dict([(x.name, i) for i, x in enumerate(self.model.layers)])

        self.train(n_epochs=n_epochs)

    def train(self, n_epochs, callbacks=[]):
        self.docvecs = None
        self.wordvecs = None
        if self.generator is None:
            self.history = self.model.fit([self.corpus.input_docs, self.corpus.input_tokens], [self.corpus.outputs],
                                          batch_size=self.params['BS'], verbose=1, epochs=n_epochs, callbacks=callbacks)
        else:
            self.history = self.model.fit_generator(self.generator,
                                                    steps_per_epoch=self.corpus.n_words * (1 + self.params['NS']) //
                                                                    self.params['BS'], initial_epoch=0, epochs=n_epochs,
                                                    verbose=1, callbacks=callbacks)

    def save(self, filename):
        json.dump(self.corpus.idx2token, open("%s.vocab" % filename, 'w'))  # Save token index mapping
        json.dump(self.params, open("%s.params" % filename, 'w'))  # Save Hyperparameters
        self.model.save(filename)

    def load(self, filename):
        self.idx2token = json.load(open("%s.vocab" % filename))  # Load token index mapping
        self.token2idx = {t: i for i, t in self.idx2token.items()}
        self.params = json.load(open("%s.params" % filename))  # Load Hyperparameters
        self.model = load_model(filename, custom_objects={'fmeasure': fmeasure})
        self.layer_lookup = dict([(x.name, i) for i, x in enumerate(self.model.layers)])

    def get_docvecs(self, min_zero=True):
        if self.docvecs is None:
            self.docvecs = self.model.layers[self.layer_lookup['docvecs']].get_weights()[0]
            if min_zero:  # Faster without relufying
                self.docvecs = relufy(self.docvecs)
        return self.docvecs

    def get_wordvecs(self, min_zero=True):
        try:
            if self.wordvecs is None:
                self.wordvecs = self.model.layers[self.layer_lookup['wordvecs']].get_weights()[0]
                if min_zero:
                    self.wordvecs = relufy(self.wordvecs)
            return self.wordvecs
        except KeyError:
            # For dense projection layer (obsolete)
            _, n_topics = self.model.layers[self.layer_lookup['docvecs']].get_weights()[0].shape
            vocab_len, word_dim = self.model.layers[self.layer_lookup['wordemb']].get_weights()[0].shape
            inlayerW = Input((1,))
            embW = Embedding(vocab_len, word_dim, input_length=1,
                             weights=self.model.layers[self.layer_lookup['wordemb']].get_weights())(inlayerW)
            embWa = Dense(n_topics, activation='relu',
                          weights=self.model.layers[self.layer_lookup['wordproj']].get_weights())(embW)
            wordvec_model = Model(inputs=[inlayerW], outputs=[embWa])
            self.wordvecs = np.reshape(wordvec_model.predict(list(range(vocab_len))), (vocab_len, n_topics))
            return self.wordvecs

    def get_topic_words(self, top_n=10, stopwords=set()):
        self.get_wordvecs()
        topic_words = {}
        for topic in range(self.wordvecs.shape[1]):
            topic_words[topic] = heapq.nlargest(top_n + len(stopwords), enumerate(L1normalize(self.wordvecs[:, topic])),
                                                key=lambda x: x[1])
            topic_words[topic] = [(self.idx2token[str(idx)], score) for idx, score in topic_words[topic] if
                                  self.idx2token[str(idx)] not in stopwords]
        self.topic_words = topic_words
        return topic_words

    def print_topic_words(self, top_n=10, stopwords=set()):
        if self.topic_words is None:
            self.get_topic_words(top_n=top_n, stopwords=stopwords)
        print("Topic words")
        for topic in self.topic_words:
            print("%d:" % topic, ', '.join(["%s" % word for word, score in self.topic_words[topic]]))

    def most_similar_words(self, word, n=20):
        self.get_wordvecs()
        idx = self.token2idx[word]
        sims = heapq.nlargest(n, enumerate(cosine_similarity(self.wordvecs[idx:idx + 1, :], self.wordvecs)[0]),
                              key=lambda x: x[1])
        return [(self.idx2token[i], s) for i, s in sims]

    def get_document_topics(self, doc_id, as_vector=False):
        """ Provide topic assignments for a document with (pseudo)probability scores """
        assignments = L1normalize(self.get_docvecs()[doc_id, :])
        if as_vector:
            return assignments  # Vector of length N_topics
        else:
            return sorted(
                filter(lambda x: x[1] > 0, enumerate(assignments)),
                key=lambda x: -1 * x[1])  # descending list of (doc_id, score)

    def get_topic_documents(self, topic_id, top_n=10):
        """ Provide most representative documents for a topic with (pseudo)probability assignment scores as in get_document_topics() """
        L1norm = np.linalg.norm(self.get_docvecs(), 1, axis=1)
        return sorted(
            filter(lambda x: x[1] > 0,
                   enumerate((self.docvecs.transpose() / L1norm)[topic_id, :])
                   ),
            key=lambda x: -1 * x[1])[:top_n]


class Logger:
    def __init__(self, filename, model, evaluator):
        self.filename = filename
        self.evaluator = evaluator
        self.model = model
        self.log = dict([('p%s' % p, v) for p, v in model.params.items()])

    def record(self, epoch, logs):
        self.log['_Epoch'] = epoch
        self.log['_Loss'] = logs['loss']
        self.log['_F1'] = logs['fmeasure']
        self.log.update(self.evaluator(self.model))
        self.write()

    def write(self):
        file_exists = isfile(self.filename)
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.log)


def data_feeder(corpus, n_passes=1, batch_size=1024 * 6):
    """ Prepare training data and vocabulary mappings from documents on the fly """
    input_docs, input_tokens, outputs = [], [], []
    # for pass_ in range(n_passes):
    pass_ = 0
    while True:
        print("\nStarting pass %d over data.\n" % (pass_ + 1))
        pass_ += 1
        random.shuffle(corpus.docs)
        for doc_id, tokens in enumerate(corpus.docs):
            # if doc_id % 100 == 0:
            #	print("\rPreparing data: %d%%" % ((doc_id+1)/len(corpus.docs)*100+1), end='', flush=True)
            # Filter tokens by frequency and map them to IDs (creates mapping table on the fly)
            token_ids = [corpus.token2idx[token] for token in tokens if corpus.cntr[token] > corpus.min_count]
            for i, idx in enumerate(token_ids):
                input_docs.append(doc_id)
                input_tokens.append(idx)
                outputs.append(1)
                input_docs.append(doc_id)
                input_tokens.append(np.random.randint(0, corpus.vocab_size - 1, 1))
                outputs.append(0)
                # if len(input_tokens) >= batch_size/(1+corpus.ns_rate) or (doc_id == len(corpus.docs)-1 and i == len(token_ids)-1):
                if len(input_tokens) >= batch_size or (doc_id == len(corpus.docs) - 1 and i == len(token_ids) - 1):
                    # Online negative sampling
                    """outputs += [0]*corpus.ns_rate*len(input_tokens)
                    input_docs += input_docs*corpus.ns_rate
                    input_tokens += list(np.random.randint(0, corpus.vocab_size-1, corpus.ns_rate*len(input_tokens)))"""
                    # Convert format
                    batch = np.concatenate([input_docs, input_tokens, outputs]).reshape(3, len(input_docs)).transpose()
                    # np.random.shuffle(batch)
                    batch = np.array(batch, dtype="int32")
                    """input_docs = np.array(input_docs, dtype="int32")
                    input_tokens = np.array(input_tokens, dtype="int32")
                    outputs = np.array(outputs, dtype="int32")"""
                    # Shuffle batch
                    """z = list(zip(*(input_docs, input_tokens, outputs)))
                    random.shuffle(z)
                    input_docs, input_tokens, outputs = map(list, zip(*z))"""
                    yield [batch[:, 0], batch[:, 1]], batch[:, 2]
                    # yield [input_docs, input_tokens], outputs
                    input_docs, input_tokens, outputs = [], [], []
