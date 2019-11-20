from flask import Flask, jsonify, request, abort, make_response
from corpus import retrieve_prepare_subtitles, retrieve_prepare_tags
from models import tfidf, lda, lftm

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
	return make_response(jsonify({'error': 'Not found'}), 404)

#################################################################
#							PREDICTION							#
#################################################################

@app.route('/api/tfidf/predict/', methods=['POST'])
def extract_topics_tfidf():
	# Extract request body parameters
	url = request.json['url']
	# Retrieve subtitles
	subtitles = retrieve_prepare_subtitles(url)
	if 'not found' == subtitles:
		return jsonify({'error':'video could not be retreived'})
	# Load the model
	model = tfidf()
	# Perform Inference
	results = model.predict(subtitles)
	# Retrieve tags
	tags = retrieve_prepare_tags(url)
	# Evaluate
	score = model.evaluate(results, tags)
	# Return resutls and score
	return jsonify({'results':results, 'score':score})

@app.route('/api/lda/predict', methods=['POST'])
def extract_topics_lda():
	# Extract request body parameters
	subtitles = retrieve_prepare_subtitles(request.json['url'])
	if 'not found' == subtitles:
		return jsonify({'error':'video could not be retreived'})
	# Load the model
	model = lda()
	# Perform Inference
	results = model.predict(subtitles)
	# Return results
	return jsonify(results)

@app.route('/api/lftm/predict', methods=['POST'])
def extract_topics_lftm():
	# Extract request body parameters
	subtitles = retrieve_prepare_subtitles(request.json['url'])
	if 'not found' == subtitles:
		return jsonify({'error':'video could not be retreived'})
	# Load the model
	model = lftm()
	# Perform Inference
	results = model.predict(subtitles)
	# Return results
	return jsonify(results)

#################################################################
#							TAGS								#
#################################################################

@app.route('/api/tags', methods=['POST'])
def extract_tags():
	# Retrieve tags
	tags = retrieve_prepare_tags(request.json['url'])
	# Return results
	return jsonify({'tags':tags})

#################################################################
#							TOPICS								#
#################################################################

@app.route('/api/lda/topics', methods=['GET'])
def get_topics_lda():
	# Load the model
	model = lda()
	# Retrieve topics
	topics = model.topics()
	# Return results
	return jsonify({'topics': topics})

@app.route('/api/lftm/topics', methods=['GET'])
def get_topics_lftm():
	# Load the model
	model = lftm()
	# Retrieve topics
	topics = model.topics()
	# Return results
	return jsonify({'topics': topics})


#################################################################
#							TRAINING							#
#################################################################

@app.route('/api/tfidf/train', methods=['POST'])
def train_tfidf():
	# Load model
	model = tfidf()
	# Train model
	results = model.train(request.json['datapath'],
		(int(request.json['min_ngram']),
		int(request.json['max_ngram'])),
		float(request.json['max_df']),
		float(request.json['min_df']))
	# return result
	return jsonify({'result':results})

@app.route('/api/lda/train', methods=['POST'])
def train_lda():
	# Load model
	model = lda()
	# Train model
	results = model.train(request.json['datapath'],
		int(request.json['num_topics']),
		int(request.json['alpha']),
		int(request.json['iterations']),
		int(request.json['optimize_interval']),
		float(request.json['topic_threshold']))
	# Return results
	return jsonify({'result':results})

@app.route('/api/lftm/train', methods=['POST'])
def train_lftm():
	# Load model
	model = lftm()
	# Train model
	results = model.train(request.json['datapath'],
		request.json['ntopics'],
		request.json['alpha'],
		request.json['beta'],
		request.json['lambda'],
		request.json['initer'],
		request.json['niter'],
		request.json['topn'])
	# Return results
	return jsonify({'result':results})

#################################################################
#							EVALUATION							#
#################################################################

@app.route('/api/lda/eval', methods=['GET'])
def eval_lda():
	# Load model
	model = lda()
	# Evaluate model
	score = model.evaluate()
	# Retrun results
	return jsonify({'score':score})

@app.route('/api/lftm/eval', methods=['GET'])
def eval_lftm():
	# Load model
	model = lftm()
	# Evaluate model
	score = model.evaluate()
	# Return results
	return jsonify({'score':score})

if __name__ == '__main__':
	app.run(debug=True)