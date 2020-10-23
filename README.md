# Topic Modeling API

This API is built to dynamically perform training, inference, and evaluation for different topic modeling techniques.
The API grant common interfaces and command for accessing the different models, make easier to compare them.

A demo is available at http://hyperted.eurecom.fr/topic.

## Models

In this repository, we provide:

* Code to perform training, inference, and evaluation for 9 Topic Modeling packages:
  * LDA from the [Mallet](http://mallet.cs.umass.edu/) package.
  * [LFTM](https://github.com/datquocnguyen/LFTM) - [paper](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158).
  * [Doc2Topic](https://github.com/sronnqvist/doc2topic)
  * [GSDMM](https://github.com/rwalk/gsdmm) - [paper](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf)
  * [Non-Negative Matrix factorization (NMF)](https://radimrehurek.com/gensim/models/nmf.html) 
  * [Hierarchical Dirichlet Processing Model (HDP)](https://radimrehurek.com/gensim/models/hdpmodel.html) 
  * [Latent Semantic Indexing (LSI)](https://radimrehurek.com/gensim/models/lsimodel.html)
  * [Paragraph Vector Topic Model (PVTM)](https://github.com/davidlenz/pvtm) - [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226685)
  * [Context Topic Model (CTM)](https://github.com/MilaNLProc/contextualized-topic-models) - [paper](https://arxiv.org/abs/2004.03974)
* A set of pre-trained models, downloadable from [here](https://www.dropbox.com/sh/sc0ffz1sig3ii5b/AAAWlM4DMpWMy2MN3CGKbWjwa?dl=0). **NOTE: Newly trained models are by default stored in `.\models`, replacing the old ones, unless a new model path is given **
* Data files containing pre-processed corpus:
  * `20ng.txt` and `20ng_labels.txt`, with 11314 news from the [20 NewsGroup dataset](http://qwone.com/~jason/20Newsgroups/)
  * `ted.txt` with 51898 subtitles of [TED Talks](https://www.ted.com/)
  * `test.txt` and `test_labels.txt`, an extraction of 30 documents from `20_ng.txt`, used for testing reason

Each model expose the following functions:

##### Training the model
```python    
m.train(data, num_topics, preprocessing) # => 'success'
```

##### Print the list of computed topics
```python
for i, x in enumerate(m.topics):
    print(f'Topic {i}')
    for word, weight in zip(x['words'], x['weights']):
        print(f'- {word} => {weight}')
```

##### Access to the info about a specific topic

```python
x = m.topic(0)
words = x['words']
weights= x['weights']
```

##### Access to the predictions computed on the training corpus

```python
for i, p in enumerate(m.get_corpus_predictions(topn=3)): # predictions for each document
    print(f'Predictions on document {i}')
    for topic, confidence in p:
        print(f'- Topic {topic} with confidence {confidence}')
        # - Topic 21 with confidence 0.03927058187976461
```

##### Predict the topic of a new text

```python
pred = m.predict(text, topn=3)
for topic, confidence in pred:
    print(f'- Topic {topic} with confidence {confidence}')
     # - Topic 21 with confidence 0.03927058187976461
```

##### Computing the coherence against a corpus

```python
# coherence: Type of coherence to compute, among <c_v, c_npmi, c_uci, u_mass>. See https://radimrehurek.com/gensim/models/coherencemodel.html#gensim.models.coherencemodel.CoherenceModel
pred = m.coherence(mycorpus, metric='c_v')
print(pred)
#{
#  "c_v": 0.5186710138972105,
#  "c_v_std": 0.1810477961008996,
#  "c_v_per_topic": [
#    0.5845048872767505,
#    0.30693460230781777,
#    0.2611738203246824,
#    ...
#  ]
#}
```

##### Evaluating against a grount truth

```python
# metric: Metric for computing the evaluation, among <purity, homogeneity, completeness, v-measure, nmi>.
res = m.get_corpus_predictions(topn=1)
v = m.evaluate(res, ground_truth_labels, metric='purity')
# 0.7825333630516738
```

The possible parameters can differ depending on the model.

## Use in a Python enviroment

Install this package

    pip install tomodapi

Use it in a Python script

```python
from tomodapi import LdaModel

# init the model 
m = LdaModel(model_path=path_location) 
# train on a corpus
m.train(my_corpus, preprocessing=False, num_topics=10) 
# infer topic of a sentence
best_topics = m.predict("In the time since the industrial revolution the climate has increasingly been affected by human activities that are causing global warming and climate change") 
topic,confidence = best_topics[0] 
# get top words for a given topic
print(m.topic(topic)) # 
```

If the `model_path` is not specified, the library will load/save the model from/under `models/<model_name>`.

## Web API

A web API is provided for accessing to the library as a service

##### Install dependencies

You should install 2 dependencies:
- [mallet 2.0.8](http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz) to be placed in `app\builtin`
- [glove.6B.50d.txt](http://nlp.stanford.edu/data/glove.6B.zip) to be placed in `app\builtin\glove`

Under UNIX, you can use the **download_dep.sh** script.

    sh download_dep.sh


##### Start the server

    python server.py

#### Docker

Alternatively, you can run a docker container with

    docker-compose -f docker-compose.yml up

The container uses **mounted volumes** so that you can easily update/access to the computed models and the data files.

#### Manual Docker installation

    docker build -t hyperted/topic .
    docker run -p 27020:5000 --env APP_BASE_PATH=http://hyperted.eurecom.fr/topic/api -d -v /home/semantic/hyperted/tomodapi/models:/models -v /home/semantic/hyperted/tomodapi/data:/data --name hyperted_topic hyperted/topic

    # Uninstall
    docker stop hyperted_topic
    docker rm hyperted_topic
    docker rmi hyperted/topic


# Publications

If you find this library or API useful in your research, please consider citing our [paper](http://www.eurecom.fr/fr/publication/6371/download/data-publi-6371_1.pdf):

```
@inproceedings{Lisena:NLPOSS2020,
   author = {Pasquale Lisena and Ismail Harrando and Oussama Kandakji and Raphael Troncy},
   title =  {{ToModAPI: A Topic Modeling API to Train, Use and Compare Topic Models}},
   booktitle = {2$^{nd}$ International Workshop for Natural Language Processing Open Source Software (NLP-OSS)},
   year =   {2020}
}
```
