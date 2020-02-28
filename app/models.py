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
from modules.gsdmm import MovieGroupProcess
import warnings




# Function to remove specified tokens from a string
def remove_tokens(x, tok2remove):
	return ' '.join(['' if t in tok2remove else t for t in x.split()])

# Term Frequency - Inverse Document Frequency
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
		results= []
		for idx in range(len(feature_vals)):
			# results[feature_vals[idx]]=score_vals[idx]
			results.append(feature_vals[idx]);

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

	# Evaluate the top words across TED tags
	def evaluate(self, topwords, tags):
		'''
			topwords: top words and their scores from tfidf
			tags: video tags
		'''

		# Load a KeyedVector model using a pre-trained word2vec
		word2vecmodel = gensim.models.KeyedVectors.load('/app/data/word2vec.bin', mmap='r')
		# Load vocabulary
		vocab = word2vecmodel.wv.vocab
		
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

# Latent Dirichlet Allocation
class lda:

	def __init__(self):
		self.model = None

	# Load saved model
	def load(self):
		with open('/app/models/lda/lda.pkl', "rb") as input_file:
			self.model = pickle.load(input_file)
		self.model.mallet_path = '/app/modules/mallet-2.0.8/bin/mallet'
		self.model.prefix = '/app/models/mallet-dep/'

	# Perform Inference
	def predict(self, doc, topn = 5):
		# Transform document into BoW
		doc = doc.split()
		common_dictionary = self.model.id2word
		doc = common_dictionary.doc2bow(doc)
		# Get topic distribution
		doc_topic_dist = self.model[doc]
		# Sort to get the top n topics
		# sorted_doc_topic_dist = 
		# Structure the results into a dictionary
		results = [{topic:weight} for topic, weight in sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:topn]]
		# Return results
		return results

	# Train the model
	def train(self,
		datapath = '/app/data/data.txt',
		num_topics = 35,
		alpha = 50,
		random_seed = 5,
		iterations = 500,
		optimize_interval=10,
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
		id2word.filter_n_most_frequent(20)
		
		corpus = [id2word.doc2bow(doc) for doc in tokens]

		# Train the model
		mallet_path = '/app/modules/mallet-2.0.8/bin/mallet'
		prefix = '/app/models/mallet-dep/'
		self.model = gensim.models.wrappers.LdaMallet(mallet_path,
			corpus=corpus,
			num_topics=num_topics,
			alpha=alpha,
			id2word=id2word,
			random_seed = random_seed,
			prefix=prefix,
			iterations = iterations,
			optimize_interval=optimize_interval,
			topic_threshold=topic_threshold)

		# Save the model
		with open('/app/models/lda/lda.pkl', 'wb') as output:
			pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

		return 'success'

	# Get coherence of model topics
	def coherence(self, datapath = '/app/data/data.txt'):

		# Load data
		text = []
		with open(datapath,"r+") as datafile:
			while True:
				vec = datafile.readline()

				if not vec:
					break

				text.append(vec.split())

		print("Docs Loaded.")

		json_topics = {}

		topic_words = []

		for i in range(0, self.model.num_topics):

			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = {}
			topic_words.append([])
			for word, weight in self.model.show_topic(i, topn = 10):
				json_topics[str(i)]['words'][word] = weight
				topic_words[-1].append(word)

		

		dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

		print("Dictionary Created.")

		print(topic_words)

		while True:
			try:

				coherence_model = gensim.models.coherencemodel.CoherenceModel(topics = topic_words, texts = text, dictionary = dictionary, coherence='c_v')
				print("Coherence Created.")

				coherence_per_topic = coherence_model.get_coherence_per_topic()

				print("Coherence Computed")

				for i in range(len(topic_words)):
					json_topics[str(i)]['c_v'] = coherence_per_topic[i]

				json_topics['c_v'] = np.nanmean(coherence_per_topic)
				json_topics['c_v_std'] = np.nanstd(coherence_per_topic)

				break

			except KeyError as e:
				key = str(e)[1:-1]
				print(key)
				for i in range(len(topic_words)):
					if key in topic_words[i]:
						topic_words[i].remove(key)
		print(json_topics)
		return json_topics

	# Get topic-word distribution
	def topics(self):

		json_topics = {}

		topic_words = []

		for i in range(0, self.model.num_topics):

			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = []
			for word, weight in self.model.show_topic(i, topn = 10):
				json_topics[str(i)]['words'].append(word)

		return json_topics

	# Get weighted similarity of topic words and tags
	def evaluate(self, datapath = '/app/data/data.txt', tagspath = '/app/data/tags.txt', topn=5):

		# Load a KeyedVector model using a pre-trained word2vec
		word2vecmodel = gensim.models.KeyedVectors.load('/app/data/word2vec.bin', mmap='r')
		# Load vocabulary
		vocab = word2vecmodel.wv.vocab

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

# Latent Feature Topic Model
class lftm:

	def __init__(self):
		# Model is loaded as text files and jar files
		pass

	# Perform Inference
	def predict(self,
		doc,
		initer = 500,
		niter = 0,
		topn = 10,
		name = 'TEDLFLDAinf'):
		'''
			doc: the document on which to make the inference
			initer: initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component
			niter: sampling iterations for the latent feature topic models
			topn: number of the most probable topical words
			name: prefix of the inference documents
		'''

		with open('/app/data/glovetokens.pkl', "rb") as input_file:
			glovetokens = pickle.load(input_file)

		doc = ' '.join([word for word in doc.split() if word in glovetokens])

		file = open('/app/data/lftm/doc.txt', "w")
		file.write(doc)
		file.close()

		
		# Perform Inference
		completedProc = subprocess.run('java -jar /app/modules/lftm/jar/LFTM.jar -model {} -paras {} -corpus {} -initers {} -niters {} -twords {} -name {} -sstep {}'.format(
			'LFLDAinf',
			'/app/data/lftm/TEDLFLDA.paras',
			'/app/data/lftm/doc.txt',
			str(initer),
			str(niter),
			str(topn),
			name,
			'0'), shell = True)

		# os.system('mv /app/data/TEDLFLDAinf.* /app/models/lftm/')

		print(completedProc.returncode)

		file = open('/app/data/lftm/TEDLFLDAinf.theta', "r")
		doc_topic_dist = file.readline()
		file.close()

		doc_topic_dist = [(topic,float(weight)) for topic, weight in enumerate(doc_topic_dist.split())]
		sorted_doc_topic_dist = sorted(doc_topic_dist, key=lambda kv: kv[1], reverse=True)[:5]
		results = [{topic:weight} for topic, weight in sorted_doc_topic_dist]
		return results

	# Train the model
	def train(self,
		datapath = '/app/data/data.txt',
		ntopics = 35,
		alpha = 0.1,
		beta = 0.1,
		_lambda = 1,
		initer = 50,
		niter = 5,
		topn = 10):
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

		file = open('/app/data/lftm/data_glove.txt', "w")
		for doc in text:
			file.write(doc+'\n') 
		file.close()


		completedProc = subprocess.run('java -jar /app/modules/lftm/jar/LFTM.jar -model {} -corpus {} -vectors {} -ntopics {} -alpha {} -beta {} -lambda {} -initers {} -niters {} -twords {} -name {} -sstep {}'.format(
			'LFLDA',
			'/app/data/lftm/data_glove.txt',
			'/app/data/glove.6B.50d.txt',
			str(ntopics),
			str(alpha),
			str(beta),
			str(_lambda),
			str(initer),
			str(niter),
			str(topn),
			'TEDLFLDA',
			'0'), shell = True)

		print(completedProc.returncode)

		return 'success'

	# Get coherence of model topics
	def coherence(self, datapath = '/app/data/data.txt'):

		file1 = open('/app/data/lftm/TEDLFLDA.topWords')
		json_topics = {}
		topics = []
		i = 0
		while True:
			words = file1.readline()
			if not words:
				break
			if words !='\n':
				json_topics[str(i)] = {}
				json_topics[str(i)]['words'] = [w for w in words[words.index(':')+2:].split()]
				topics.append(json_topics[str(i)]['words'])
				i+=1

		text = []
		with open(datapath,"r+") as f:
			for vec in f:

				text.append(vec.split())


		print("Docs Loaded.")

		dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

		print("Dictionary Created.")

		print(topics)

		while True:
			try:

				coherence_model = gensim.models.coherencemodel.CoherenceModel(topics = topics, texts = text, dictionary = dictionary, coherence='c_v')
				print("Coherence Created.")

				coherence_per_topic = coherence_model.get_coherence_per_topic()

				print("Coherence Computed")

				for i in range(len(topics)):
					json_topics[str(i)]['c_v'] = coherence_per_topic[i]

				json_topics['c_v'] = np.nanmean(coherence_per_topic)
				json_topics['c_v_std'] = np.nanstd(coherence_per_topic)

				break

			except KeyError as e:
				key = str(e)[1:-1]
				print(key)
				for i in range(len(topics)):
					if key in topics[i]:
						topics[i].remove(key)
		print(json_topics)
		return json_topics

	# Get topic-word distribution
	def topics(self):

		file1 = open('/app/data/lftm/TEDLFLDA.topWords')
		json_topics = {}
		i = 0
		while True:
			words = file1.readline()
			if not words:
				break
			if words !='\n':
				json_topics[str(i)] = {}
				json_topics[str(i)]['words'] = [w for w in words[words.index(':')+2:].split()]
				i+=1

		return json_topics


	# Get weighted similarity of topic words and tags
	def evaluate(self, tagspath = '/app/data/tags.txt', topn = 5):

		# Load a KeyedVector model using a pre-trained word2vec
		word2vecmodel = gensim.models.KeyedVectors.load('/app/data/word2vec.bin', mmap='r')
		# Load vocabulary
		vocab = word2vecmodel.wv.vocab

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

# Neural Topic Model
class ntm:

	def __init__(self):
		pass

	# Load the saved model
	def load(self):
		self.model = models.Doc2Topic()
		self.model.load(filename = '/app/models/ntm/ntm')

	# Get coherence of model topics
	def coherence(self, datapath = '/app/data/data.txt'):

		# Load data
		text = []
		with open(datapath,"r+") as datafile:
			while True:
				vec = datafile.readline()

				if not vec:
					break

				text.append(vec.split())

		print("Docs Loaded.")

		topics = self.model.get_topic_words()
		
		json_topics = {}

		topic_words = []

		for i, topic in topics.items():
			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = {}
			topic_words.append([])
			for word, weight in topic:
				json_topics[str(i)]['words'][word] = float(weight)
				topic_words[-1].append(word)


		dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

		print("Dictionary Created.")
		
		print(topic_words)
		
		while True:
			try:

				coherence_model = gensim.models.coherencemodel.CoherenceModel(topics = topic_words, texts = text, dictionary = dictionary, coherence='c_v')
				print("Coherence Created.")

				coherence_per_topic = coherence_model.get_coherence_per_topic()

				print("Coherence Computed")

				for i in range(len(topics)):
					json_topics[str(i)]['c_v'] = coherence_per_topic[i]

				json_topics['c_v'] = np.nanmean(coherence_per_topic)
				json_topics['c_v_std'] = np.nanstd(coherence_per_topic)

				print("Coherence Computed")

				break

			except KeyError as e:
				key = str(e)[1:-1]
				print(key)
				for i in range(len(topic_words)):
					if key in topic_words[i]:
						topic_words[i].remove(key)
		print(json_topics)
		return json_topics

	# Get topic-word distribution
	def topics(self):

		topics = self.model.get_topic_words()
		
		json_topics = {}

		
		for i, topic in topics.items():
			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = {}
			for word, weight in topic:
				json_topics[str(i)]['words'][word] = float(weight)

		return json_topics

	# Train the model
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
# Gibbs Sampling Algorithm for a Dirichlet Mixture Model
class gsdmm:

	def __init__(self):
		pass

	# Load the saved model
	def load(self):
		with open('/app/models/gsdmm/gsdmm.pkl', "rb") as input_file:
			self.model = pickle.load(input_file)

	# Get coherence of model topics
	def coherence(self, datapath = '/app/data/data.txt'):

		# Load data
		text = []
		with open(datapath,"r+") as datafile:
			while True:
				vec = datafile.readline()

				if not vec:
					break

				text.append(vec.split())

		print(" Docs Loaded.")

		json_topics = {}

		topic_words = []

		for i, topic in enumerate(self.model.cluster_word_distribution):

			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = {}
			topic_words.append([])
			total = sum(topic.values())
			for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse = True)[:10]:
				json_topics[str(i)]['words'][word] = freq/total
				topic_words[-1].append(word)

		

		dictionary = gensim.corpora.hashdictionary.HashDictionary(text)

		print("Dictionary Created.")

		print(topic_words)

		while True:
			try:

				coherence_model = gensim.models.coherencemodel.CoherenceModel(topics = topic_words, texts = text, dictionary = dictionary, coherence='c_v')
				print("Coherence Created.")

				coherence_per_topic = coherence_model.get_coherence_per_topic()

				print("Coherence Computed")

				for i in range(len(topic_words)):
					json_topics[str(i)]['c_v'] = coherence_per_topic[i]

				json_topics['c_v'] = np.nanmean(coherence_per_topic)
				json_topics['c_v_std'] = np.nanstd(coherence_per_topic)

				break

			except KeyError as e:
				key = str(e)[1:-1]
				print(key)
				for i in range(len(topic_words)):
					if key in topic_words[i]:
						topic_words[i].remove(key)
		print(json_topics)
		return json_topics

	# Get topic-word distribution
	def topics(self):

		json_topics = {}

		for i, topic in enumerate(self.model.cluster_word_distribution):

			json_topics[str(i)] = {}
			json_topics[str(i)]['words'] = []
			total = sum(topic.values())
			for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse = True)[:10]:
				# json_topics[str(i)]['words'][word] = freq/total
				json_topics[str(i)]['words'].append(word)

		return json_topics

	# Train the model
	def train(self,
		datapath = '/app/data/data.txt',
		n_topics=35, 
		alpha=0.1,
		beta=0.1, 
		n_iter=15):

		# Build the model
		self.model = MovieGroupProcess(K=n_topics, alpha=alpha, beta=beta, n_iters=n_iter)

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

		# Fit the model
		self.model.fit(tokens, len(id2word))

		# Save the new model
		with open('/app/models/gsdmm/gsdmm.pkl', 'wb') as output:
			pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)

		return 'success'

	# Perform Inference
	def predict(self, doc):
		results = [(topic,score) for topic, score in enumerate(self.model.score(doc))]
		results = [{topic:weight} for topic, weight in sorted(results, key=lambda kv: kv[1], reverse=True)[:5]]
		return results



