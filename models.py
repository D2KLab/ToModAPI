import pickle
import sklearn
import gensim
import os
import subprocess
import numpy as np
import requests
from threading import Semaphore
from scipy.sparse import coo_matrix
from modules.doc2topic import models, corpora
import warnings




# Function to remove specified tokens from a string
def remove_tokens(x, tok2remove):
	return ' '.join(['' if t in tok2remove else t for t in x.split()])

# Load a KeyedVector model using a pre-trained word2vec
word2vecmodel = gensim.models.KeyedVectors.load('/app/data/word2vec.bin', mmap='r')
# Load vocabulary
vocab = word2vecmodel.wv.vocab


class tfidf:

	def __init__(self):
		self.model = None

	# Load saved model
	def load(self):
		with open('/app/models/tfidf/tfidf.pkl', "rb") as input_file:
			self.model = pickle.load(input_file)

	# Perform Inference
	def predict(self, doc, topn = 5):
		'''
			doc: text on which to perform inference
			topn: the number of top keywords to extract
		'''

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
			score_vals.append(round(score**2, 3))
			feature_vals.append(feature_names[idx])
	
		# Create a tuples of feature,score
		# Results = zip(feature_vals,score_vals)
		results= {}
		for idx in range(len(feature_vals)):
			results[feature_vals[idx]]=score_vals[idx]

		return results

	# Train the model
	def train(self, datapath = '/app/data/data.txt', ngram_range = (1,2), max_df = 1.0, min_df = 1):
		'''
			datapath: path to training data text file
			ngram_range: the range of ngrams to consider
			max_df: the max document frequency to consider
			min_df: the min document frequency to consider
		'''
		
		text = []
		with open(datapath,"r+") as f:
			while True:
				vec = f.readline()

				if not vec:
					break

				text.append(vec)

		# Create a new model and fit it to the data
		self.model = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)  
		self.model.fit(text)

		# Save the new model
		with open('/app/models/tfidf/tfidf.pkl', 'wb') as output:
			pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

		return 'success'

	def evaluate(self, topwords, tags):
		'''
			topwords: top words and their scores from tfidf
			tags: video tags
		'''
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
							similarity= weight*word2vecmodel.similarity(word,tag)
							if similarity > max_similarity[word_id]:
								max_similarity[word_id] = similarity
				word_id += 1
				total_weights += weight
		# Compute the weighted mean
		if word_id > 0:
			score = sum(max_similarity)
			score /= total_weights

		return score

class lda:

	def __init__(self):
		self.model = None

	# Load saved model
	def load(self):
		with open('/app/models/lda/lda.pkl', "rb") as input_file:
			self.model = pickle.load(input_file)
		self.model.mallet_path = '/app/modules/mallet-2.0.8/bin/mallet'
		self.model.prefix = '/app/modules/mallet-dep/'

	# Perform Inference
	def predict(self, doc, topn = 5):
		# Transform document into BoW
		doc = doc.split()
		common_dictionary = self.model.id2word
		doc = common_dictionary.doc2bow(doc)
		# Get topic distribution
		doc_topic_dist = self.model[doc]
		# Sort to get the top n topics
		sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]
		# Structure the results into a dictionary
		results = {topic:weight for topic, weight in sorted_doc_topic_dist}
		# Return results
		return results

	# Train the model
	def train(self,
		datapath = '/app/data/data.txt',
		num_topics = 35,
		alpha = 50,
		iterations = 500,
		optimize_interval=0,
		topic_threshold=0.0):
		'''
			datapath: path to training data text file
			num_topics: number of topics
			alpha: prior document-topic distribution
			iternations: number of iteration in EM
			optimize_interval: hyperparameter optimization every optimize_interval
			topic_threshold:  threshold of the probability above which we consider a topic
		'''


		# Load data
		text = []
		with open(datapath,"r+") as datafile:
			while True:
				vec = datafile.readline()

				if not vec:
					break

				text.append(vec)

		# Transform documents
		tokens = [doc.split() for doc in text]

		id2word = gensim.corpora.Dictionary(tokens)
		
		corpus = [id2word.doc2bow(doc) for doc in tokens]

		# Train the model
		mallet_path = '/app/modules/mallet-2.0.8/bin/mallet'
		prefix = '/app/modules/mallet-dep/'
		self.model = gensim.models.wrappers.LdaMallet(mallet_path,
			corpus=corpus,
			num_topics=num_topics,
			alpha=alpha,
			id2word=id2word,
			random_seed = 1,
			prefix=prefix,
			iterations = iterations,
			optimize_interval=optimize_interval,
			topic_threshold=topic_threshold)

		# Save the model
		with open('/app/models/lda/lda.pkl', 'wb') as output:
			pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

		return 'success'

	# Get topic-word distribution
	def topics(self):

		json_topics = {}

		for i in range(0, self.model.num_topics-1):

			topic_words = []
			topic = self.model.show_topic(i, topn = 5)

			json_topics[str(i)] = {}

			for word, weight in topic:
				json_topics[str(i)][word] = weight

		return json_topics

	# Get weighted similarity of topic words and tags
	def evaluate(self, datapath = '/app/data/data.txt', tagspath = '/app/data/tags.txt', topn=5):

		# Extract and transform text
		text = []
		with open(datapath,"r+") as f:
			while True:
				vec = f.readline()

				if not vec:
					break

				text.append(vec)

		
		
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
			
			print('doc',num_doc)
			doc_score = 0
			topic_weights = 0
			# Iterate over each topic-word distribution
			for topic_id, topic_weight in sorted_doc_topic_dist:
				topic_words = self.model.show_topic(topic_id, topn = topn)
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
									similarity = weight*word2vecmodel.similarity(word,tag)
									if similarity > max_similarity[word_id]:
										max_similarity[word_id] = similarity
						

						word_id += 1
						word_weights += weight
				# Get topic score
				topic_score = sum(max_similarity)/word_weights
				# Update document score
				doc_score += topic_weight*topic_score
				topic_weights += topic_weight
			# Get document score
			doc_score /= topic_weights
			print('doc score',doc_score)

			score += doc_score
			num_doc += 1
		# Get total score for all documents
		score /= num_doc

		# Return score
		return score

class lftm:

	def __init__(self):
		# Model is loaded as text files and jar files
		pass

	def predict(self,
		doc,
		initer = 1,
		niter = 1,
		topn = 20,
		name = 'TEDLFLDAinf'):
		'''
			doc: the document on which to make the inference
			initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
			niter: sampling iterations for the latent feature topic models
			topn: number of the most probable topical words
			name: prefix of the inference documents
		'''
		file = open('/app/data/doc.txt', "w")
		file.write(doc)
		file.close()

		
		# Perform Inference
		completedProc = subprocess.run('java -jar /app/modules/lftm/jar/LFTM.jar -model {} -paras {} -corpus {} -initers {} -niters {} -twords {} -name {} -sstep {}'.format(
			'LFLDAinf',
			'/app/models/lftm/TEDLFLDA.paras',
			'/app/data/doc.txt',
			str(initer),
			str(niter),
			str(topn),
			name,
			'0'), cwd='/app/models/lftm/', shell = True)

		os.system('mv /app/data/TEDLFLDAinf.* /app/models/lftm/')

		print(completedProc.returncode)

		file = open('/app/models/lftm/TEDLFLDAinf.theta', "r")
		doc_topic_dist = file.readline()
		file.close()

		doc_topic_dist = [(topic,float(weight)) for topic, weight in enumerate(doc_topic_dist.split())]
		sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:10]
		results = {topic:weight for topic, weight in sorted_doc_topic_dist}
		return results

	def train(self,
		datapath = '/app/data/data.txt',
		ntopics = 35,
		alpha = 0.1,
		beta = 0.1,
		_lambda = 1,
		initer = 50,
		niter = 5,
		topn = 20):
		'''
			datapath: the path to the training text file
			ntopics: the number of topics
			alpha: prior document-topic distribution
			beta: prior topic-word distribution
			lambda: mixture weight
			initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
			niter: sampling iterations for the latent feature topic models
			topn: number of the most probable topical words
		'''

		with open('/app/data/glovetokens.pkl', "rb") as input_file:
			glovetokens = pickle.load(input_file)

		text = []
		with open(datapath,"r+") as f:
			while True:
				vec = f.readline()

				if not vec:
					break

				text.append(vec)

		tokens = [doc.split() for doc in text]

		id2word = list(gensim.corpora.Dictionary(tokens).values())

		
		tok2remove = {}
		for t in id2word:
			if t not in glovetokens:
				tok2remove[t] = True


		text = [remove_tokens(doc,tok2remove) for doc in text]

		file = open('/app/data/data_glove.txt', "w")
		for doc in text:
			file.write(doc+'\n') 
		file.close()


		completedProc = subprocess.run('java -jar /app/modules/lftm/jar/LFTM.jar -model {} -corpus {} -vectors {} -ntopics {} -alpha {} -beta {} -lambda {} -initers {} -niters {} -twords {} -name {} -sstep {}'.format(
			'LFLDA',
			'/app/data/data_glove.txt',
			'/app/data/glove.6B.50d.txt',
			str(ntopics),
			str(alpha),
			str(beta),
			str(_lambda),
			str(initer),
			str(niter),
			str(topn),
			'TEDLFLDA',
			'0'), cwd='/app/models/lftm/', shell = True)

		os.system('mv /app/data/TEDLFLDA.* /app/models/lftm/')

		print(completedProc.returncode)

		return 'success'

	# Extract topics
	def topics(self):
		file1 = open('/app/models/lftm/TEDLFLDA.topWords')
		topics = {}
		i = 0
		while True:
			words = file1.readline()
			if not words:
				break
			if words !='\n':
				print()
				topics[str(i)] = [w for w in words[words.index(':')+2:].split()]
				i+=1

		return topics


	# Evaluate the model on a corpus
	def evaluate(self, tagspath = '/app/data/tags.txt', topn = 5):
		file1 = open('/app/models/lftm/TEDLFLDA.topWords')
		file2 = open('/app/models/lftm/TEDLFLDA.theta')
		file3 = open(tagspath)

		# Prepare Topics
		topics = {}
		i = 0
		while True:
			words = file1.readline()
			if not words:
				break
			if words !='\n':
				topics[str(i)] = [w for w in words.split(": ",1)[1].split()][:topn]
				i+=1

		# Prepare document-topic distribution
		doc_topic_dist = []
		while True:
			topic_dist = file2.readline()
			if not topic_dist:
				break
			if topic_dist !='\n':
				doc_topic_dist.append([float(doc) for doc in topic_dist.split()][:topn])

		score = 0
		num_doc = 0
		# Irerate over each document
		while True:
			tags = file3.readline()
			if not tags:
				break

			print('doc',num_doc)
			doc_score = 0
			topic_weights = 0
			# Iterate over the top topics
			for topic_id, topic_weight in enumerate(doc_topic_dist[num_doc]):
				topic_words = topics[str(topic_id)]
				max_similarity = []
				word_id = 0
				# Iterate over topic words
				for word in topic_words:
					if word in vocab:
						max_similarity.append(0)
						# Compute the maximum similarity with tags
						for tag in tags.split(' '):
								if tag in vocab and 'ted' not in tag.lower():
									similarity = word2vecmodel.similarity(word,tag)
									if similarity > max_similarity[word_id]:
										max_similarity[word_id] = similarity
						
						word_id += 1
				topic_score = sum(max_similarity)/word_id
				doc_score += topic_weight*topic_score
				topic_weights += topic_weight
			doc_score /= topic_weights
			print('doc score',doc_score)

			score += doc_score
			num_doc += 1

		score /= num_doc

		print(score)
		return score

class ntm:

	def __init__(self):
		pass

	def load(self):
		self.model = models.Doc2Topic()
		self.model.load(filename = '/app/models/ntm/ntm')

	def topics(self):
		topics = self.model.get_topic_words()
		print(topics)
		json_topics = {}

		for i, topic in topics.items():
			json_topics[str(i)] = {}

			for word, weight in topic:
				json_topics[str(i)][word] = float(weight)

		return json_topics

	def train(self,
		datapath = '/app/data/data.txt',
		n_topics=35, 
		batch_size=1024*6,
		n_epochs=20, 
		lr=0.05, 
		l1_doc=0.000002, 
		l1_word=0.000000015, 
		word_dim=None, 
		generator=None):


		warnings.filterwarnings("ignore")

		models.init_tf_memory()

		data = corpora.DocData(datapath)

		self.model = models.Doc2Topic()

		self.model.build(data, n_topics=n_topics, batch_size=batch_size, n_epochs=n_epochs, lr=lr, l1_doc=l1_doc, l1_word=l1_word, word_dim=word_dim, generator=generator)

		fmeasure = self.model.history.history['fmeasure'][-1]
		loss = self.model.history.history['loss'][-1]

		self.model.save('/app/models/ntm/ntm')

		return 'success', fmeasure, loss