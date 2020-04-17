API Documentation
=================

**DISCLAIMER.** This documentation can be not updated. Please refer to http://hyperted.eurecom.fr/topic.


### Training

Training is performed using **POST** requests on specific endpoints:

##### **TFIDF**

On [localhost:5000/api/tfidf/train](localhost:5000/api/tfidf/train) with this body pattern:


    {
      "datapath": "/app/data/data.txt",
      "min_ngram": "1",
      "max_ngram": "2",
      "max_df": "1.0",
      "min_df": "0.01"
    }

##### **LDA**

On [localhost:5000/api/lda/train](localhost:5000/api/lda/train) with this body pattern:


    {
      "datapath": "/app/data/data.txt",
      "num_topics": "25",
      "alpha": "0.1",
      "random_seed": 5,
      "iterations": "800",
      "optimize_interval": "10",
      "topic_threshold": "0"
    }

##### **LFTM**

On [localhost:5000/api/lftm/train](localhost:5000/api/lftm/train) with this body pattern:


      {
        "datapath": "/app/data/data.txt",
        "ntopics": "25",
        "alpha": "0.1",
        "beta": "0.1",
        "lambda": "1",
        "initer": "700",
        "niter": "100",
        "topn": "10"
      }

##### **NTM**

On [localhost:5000/api/ntm/train](localhost:5000/api/ntm/train) with this body pattern:


    {
      "datapath" : "/app/data/data.txt",
      "n_topics":"25",
      "batch_size":"6144",
      "n_epochs":"50",
      "lr": "0.05",
      "l1_doc":"0.000002",
      "l1_word":"0.000000015",
      "word_dim":"0"
    }

##### **GSDMM**

On [localhost:5000/api/gsdmm/train](localhost:5000/api/gsdmm/train) with this body pattern:


    {
      "datapath": "/app/data/data.txt",
      "num_topics": "25",
      "alpha": "0.1",
      "beta": "0.1",
      "n_iter": "10"
    }

##### **TFIDF**

On [localhost:5000/api/tfidf/train](localhost:5000/api/tfidf/train) with this body pattern:


    {
      "datapath": "/app/data/data.txt",
      "min_ngram": "1",
      "max_ngram": "2",
      "max_df": "1.0",
      "min_df": "0.01"
    }


### Inference

Inference is performed using **POST** requests on specific endpoints:

[localhost:5000/api/tfidf/predict](localhost:5000/api/tfidf/predict)

[localhost:5000/api/lda/predict](localhost:5000/api/lda/predict)

**NOTE: for LFTM you need to [download](http://nlp.stanford.edu/data/glove.6B.zip) glove word vectors (specifically glove.6B.50d.txt) and store it in the data directory.**

[localhost:5000/api/lftm/predict](localhost:5000/api/lftm/predict)

[localhost:5000/api/ntm/predict](localhost:5000/api/ntm/predict)

[localhost:5000/api/gsdmm/predict](localhost:5000/api/gsdmm/predict)

with the same body:

    {
      "text": "Text Here"
    }

### Topic Distribution

To retrieve the **topic-word** distribution from each topic model. Use the **GET** requests on endpoints:

[localhost:5000/api/lda/topics](localhost:5000/api/lda/topics)

[localhost:5000/api/lftm/topics](localhost:5000/api/lftm/topics)

[localhost:5000/api/ntm/topics](localhost:5000/api/ntm/topics)

[localhost:5000/api/gsdmm/topics](localhost:5000/api/gsdmm/topics)


### Topic Distribution and Coherence Evaluation

To retrieve the **topic-word** distribution from each topic model. Use the **POST** requests on endpoints:

[localhost:5000/api/lda/coherence](localhost:5000/api/lda/coherence)

[localhost:5000/api/lftm/coherence](localhost:5000/api/lftm/coherence)

[localhost:5000/api/ntm/coherence](localhost:5000/api/ntm/coherence)

[localhost:5000/api/gsdmm/coherence](localhost:5000/api/gsdmm/coherence)

    {
      "datapath": "/app/data/data.txt"
    }

The **datapath** identifies the reference corpus text file to compute the [topic coherence](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)

### Retieve video tags

To retrieve the tags of a specific video, perform a **POST** requests on [localhost:5000/api/tags](localhost:5000/api/tags) with body:

    {
      "url": "https://www.ted.com/talks/brandon_clifford_architectural_secrets_of_the_world_s_ancient_wonders"
    }

### Evaluation on tags for the model

This API also provides the capabilities to f=compute the score based on the similarity between the topics (or words) extracted and the tags of the video. **NOTE: this is unique for videos and not chapters.** This uses pretrained word2vec to compute the average weighted similarities.

This is done using a **GET** request on:

[localhost:5000/api/lda/eval](localhost:5000/api/lda/eval)

[localhost:5000/api/lftm/eval](localhost:5000/api/lftm/eval)
