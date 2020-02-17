# Flask RESTful API for TED Talk Topic Modelling

## The purpose of this API.

This API is built to dynamically perform training, inference, and evaluation for different topic modeling techniques. It's mainly directed towards [TED](https://www.ted.com/) talks to dynamically build topic distributions for talks covering different topics.

## How this API is built.

This API uses **Python's Flask Framework** to easily let developers and users integrate and use different Topic Modeling techniques and packages. In order to perform topic extraction tasks using RESTful GET and POST requests. 

* This API can be used in development mode using mounted volumes in the **docker-compose-dev.yml** file.

## What is already provided.

In this repository, we provide:

* Pre-trained models in the **models/** directory. These are automatically used in the inference and evaluation API calls. **NOTE: Newly trained models are stored there, but replace the old ones.**
* Code to retrieve TED talk subtitles and tags.
* Code to perform training, inference, and evaluation for 4 Topic Modeling packages:
  * Latent Dirichlet allocation ([LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) from the [Mallet](http://mallet.cs.umass.edu/) package.
  * Latent Feature Topic Model ([LFTM](https://github.com/datquocnguyen/LFTM)) from Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. [Improving Topic Models with Latent Feature Word Representations](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158).
  * Neural Topic Model ([NTM](https://github.com/sronnqvist/doc2topic)).
  * Gibbs sampling algorithm for a Dirichlet Mixture Model ([GSDMM](https://github.com/rwalk/gsdmm)) of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for clustering short text documents.
  * **Extra model for keyword extraction** - Term Frequency Inverse Document Frequency ([TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) to extract keywords for documents.

* Data files containing all subtitles available on TED as well as the tags of TED videos.

## How to use the API.

### Launching

Download the mallet package by running the **download_dep.sh** script.

    sh download_dep.sh


**NOTE: you need to [download](http://nlp.stanford.edu/data/glove.6B.zip) glove word vectors (specifically glove.6B.50d.txt) and store it in the data directory in order to be able to you LFTM.**

#### Production

    docker-compose -f docker-compose.yml up

This version will copy the TED data need for training and some dependencies like Glove word vectors for LFTM.

#### Dev Mod

    docker-compose -f docker-compose-dev.yml up

This command will build the image from the Dockerfile in **app/deployment/** and start the container.

The container uses **mounted volumes** so that you can easily update the code, the model files, and the data files.

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

This API also provides the capabilities to compute the score based on the similarity between the topics (or words) extracted and the tags of the video. **NOTE: this is unique for videos and not chapters.** This uses pretrained word2vec to compute the average weighted similarities.

This is done using a **GET** request on:

[localhost:5000/api/lda/eval](localhost:5000/api/lda/eval)

[localhost:5000/api/lftm/eval](localhost:5000/api/lftm/eval)

## Credit

> [LFTM](https://github.com/datquocnguyen/LFTM) by [Dat Quoc Nguyen](https://github.com/datquocnguyen).

> [Doc2Topic/NTM](https://github.com/sronnqvist/doc2topic) (modified version) by [Samuel Rönnqvist](https://github.com/sronnqvist).

> [GSDMM](https://github.com/rwalk/gsdmm) by [Ryan Walker](https://github.com/rwalk).
