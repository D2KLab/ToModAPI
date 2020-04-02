import pickle
import sklearn
import gensim
from .abstract_model import AbstractModel

ROOT = ''
MODEL_PATH = ROOT + '/models/tfidf/tfidf.pkl'
W2V_PATH = ROOT + '/data/word2vec.bin'


# Term Frequency - Inverse Document Frequency
class TfIdfModel(AbstractModel):
    # Load saved model
    def load(self):
        with open(MODEL_PATH, "rb") as input_file:
            self.model = pickle.load(input_file)

    # Perform Inference
    def predict(self, doc, topn=5):
        """
            doc: text on which to perform inference
            topn: the number of top keywords to extract
        """
        if self.model is None:
            self.load()

        # Transform document into TFIDF
        coo_matrix = self.model.transform([doc]).tocoo()

        tuples = zip(coo_matrix.col, coo_matrix.data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

        # Get the feature names and tf-idf score of top items
        feature_names = self.model.get_feature_names()

        # Use only top items from vector
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []

        # Word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score ** 2, 3))
            feature_vals.append(feature_names[idx])

        # Create a tuples of feature,score
        # Results = zip(feature_vals,score_vals)
        results = []
        for idx in range(len(feature_vals)):
            # results[feature_vals[idx]]=score_vals[idx]
            results.append(feature_vals[idx])

        return results

    # Train the model
    def train(self, datapath='/app/data/data.txt', ngram_range=(1, 2), max_df=1.0, min_df=1):
        """
            datapath: path to training data text file
            ngram_range: the range of ngrams to consider
            max_df: the max document frequency to consider
            min_df: the min document frequency to consider
        """

        with open(datapath, "r") as datafile:
            text = [line.rstrip() for line in datafile if line]

        # Create a new model and fit it to the data
        self.model = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range,
                                                                     max_df=max_df,
                                                                     min_df=min_df)
        self.model.fit(text)

        # Save the new model
        with open(MODEL_PATH, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

        return 'success'

    # Evaluate the top words across TED tags
    def evaluate(self, topwords, tags):
        """
            topwords: top words and their scores from tfidf
            tags: video tags
        """

        # Load a KeyedVector model using a pre-trained word2vec
        word2vec = gensim.models.KeyedVectors.load(W2V_PATH, mmap='r')
        # Load vocabulary
        vocab = word2vec.wv.vocab

        score = 0
        word_id = 0
        total_weights = 0
        max_similarity = []
        # Iterate over top words
        for word, weight in topwords.items():
            # Verify that the word has a word vector
            if word in vocab:
                max_similarity.append(0)
                # Calculate the maximum similarity among the tags
                for tag in tags.split(','):
                    if tag in vocab and 'ted' not in tag.lower():
                        similarity = weight * word2vec.similarity(word, tag)
                        if similarity > max_similarity[word_id]:
                            max_similarity[word_id] = similarity
                word_id += 1
                total_weights += weight
        # Compute the weighted mean
        if word_id > 0:
            score = sum(max_similarity)
            score /= total_weights

        return score

    def coherence(self, datapath='/data/data.txt', coherence='c_v'):
        return {'message': 'not foreseen for this model'}

    def topics(self):
        return {'message': 'not foreseen for this model'}
