import re
import time

from flask import Flask, jsonify, request, make_response
from flask_restx import Api, Resource, Namespace
from flask_cors import CORS

from pydoc import locate
from docstring_parser import parse as docparse

from tomodapi.abstract_model import AbstractModel

AbstractModel.ROOT = ''
import tomodapi as models

__package__ = 'tomodapi'

app = Flask(__name__)
api = Api(app, version='1.0', title='Topic Model API', prefix='/api',
          description='''This is an API used to train, evaluate, and operate unsupervised topic models.
              Source code: [https://github.com/D2KLab/Topic-Model-API](https://github.com/D2KLab/Topic-Model-API).''')

CORS(app)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found', 'detail': str(error)}), 404)


among_regex = r"among <(.+(?:, ?.+)+)>"

model_index = {}


def extract_model_id(req):
    _, _, _model_id, _ = req.path.split('/', 3)
    return _model_id


def extract_parameter(fun):
    params = {}
    argcount = fun.__code__.co_argcount
    defaults = fun.__defaults__[0:]
    for i, p in enumerate(fun.__code__.co_varnames[argcount-len(defaults):argcount]):
        params[p] = {'default': defaults[i]}
    for p in docparse(fun.__doc__).params:
        if p.arg_name in ['datapath', 'num_topics', 'coherence', 'model']:
            params[p.arg_name]['required'] = True

        if p.arg_name in params:
            params[p.arg_name]['type'] = locate(p.type_name) if p.type_name else str
            params[p.arg_name]['description'] = p.description
            matches = re.finditer(among_regex, p.description, re.IGNORECASE)
            for m in matches:
                params[p.arg_name]['enum'] = [x.strip() for x in m.group(1).split(',')]

        if p.arg_name in ['num_topics']:
            params[p.arg_name]['type'] = int
    return params


coherence_params = extract_parameter(AbstractModel.coherence)
evaluate_params = extract_parameter(AbstractModel.evaluate)

for model in models.__all__:
    model_name = str(model.__name__).replace('Model', '').lower()
    model_index[model_name] = model
    doc = model.__doc__
    ns = Namespace(model_name, description=doc.split('\n')[0])

    train_params = extract_parameter(model.train)


    @ns.route('/train')
    class Train(Resource):
        @ns.doc(description='''Train the model''',
                params=train_params)
        def get(self):
            start = time.time()

            _model_name = extract_model_id(request)
            m = model_index[_model_name]()
            train_params = extract_parameter(m.train)
            params = [request.args.get(k, default=p['default'], type=p['type']) for k, p in train_params.items()]
            print(train_params.items())
            print(params)
            results = m.train(*params)
            m.save()
            dur = time.time() - start
            print(f'Training {_model_name} done in {dur}')
            # Return results
            return make_response(jsonify({'time': dur, 'result': results}), 200)


    @ns.route('/predict')
    class Predict(Resource):
        @ns.doc(description='Predict the topic of a text',
                params={
                    'text': {'description': 'The text on which performing the prediction',
                             'required': True,
                             'default': 'Climate change is a global environmental issue that is affecting the lands, '
                                        'the oceans, the animals, and humans'},
                    'topn': {
                        'description': 'The number of most probable topics to return',
                        'type': int, 'default': 5
                    },
                    'preprocessing': {
                        'description': 'If True, execute preprocessing on the document',
                        'type': bool, 'default': False
                    }
                },
                required=['text'])
        def get(self):
            start = time.time()

            text = request.args.get('text', type=str)
            topn = request.args.get('topn', default=5, type=int)
            m = model_index[extract_model_id(request)]()
            results = m.predict(text, topn=topn, preprocessing=True)
            dur = time.time() - start
            print(results)
            return make_response(jsonify({'time': dur, 'results': results}), 200)


    @ns.route('/corpus_prediction')
    class CorpusPrediction(Resource):
        @ns.doc(description='''Returns the predictions computed on the training corpus.
        This is not re-computing predictions, but reading training results.''',
                params={'topn': {
                    'description': 'The number of most probable topics to return.',
                    'type': int, 'default': 5
                }})
        def get(self):
            start = time.time()

            topn = request.args.get('topn', default=5, type=int)
            m = model_index[extract_model_id(request)]()
            results = m.get_corpus_predictions(topn)
            dur = time.time() - start
            return make_response(jsonify({'time': dur, 'results': results}), 200)


    @ns.route('/topics')
    class Topics(Resource):
        @ns.doc(description='''Returns the model topic list''')
        def get(self):
            start = time.time()
            m = model_index[extract_model_id(request)]()
            topics = m.topics
            dur = time.time() - start
            return make_response(jsonify({'time': dur, 'topics': topics}), 200)


    @ns.route('/topic/<id>')
    class Topic(Resource):
        @ns.doc(description='''Returns the model topic list''',
                params={'id': {'description': 'Topic id', 'required': True, 'type': int}})
        def get(self, id):
            m = model_index[extract_model_id(request)]()
            topics = m.topic(int(id))
            return make_response(jsonify(topics), 200)


    @ns.route('/coherence')
    class Coherence(Resource):
        @ns.doc(description='''Compute the coherence against a corpus''', params=coherence_params)
        def get(self):
            start = time.time()

            m = model_index[extract_model_id(request)]()
            params = [request.args.get(k, default=p['default'], type=p['type']) for k, p in coherence_params.items()]
            dur = time.time() - start
            topics = m.coherence(*params)
            topics['time'] = dur
            response = jsonify(topics)
            # os.makedirs(AbstractModel.ROOT + '/data/out', exist_ok=True)
            # output_file = AbstractModel.ROOT + '/data/out/%s.%s.json' % (
            #     model_name, request.args.get('coherence', default='c_v'))
            # with open(output_file, 'w') as f:
            #     json.dump(response, f)
            return make_response(response, 200)


    @ns.route('/evaluate')
    class Evaluate(Resource):
        @ns.doc(description='''Evaluate against a ground truth''', params=evaluate_params)
        def get(self):
            start = time.time()

            m = model_index[extract_model_id(request)]()
            params = [request.args.get(k, default=p['default'], type=p['type']) for k, p in evaluate_params.items()]
            dur = time.time() - start
            result = m.evaluate(*params)
            response = jsonify({
                'time': dur,
                'result': result
            })
            return make_response(response, 200)


    api.add_namespace(ns)

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0')
