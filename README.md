# Flask RESTful API for Topic Modelling

This API is built to dynamically perform training, inference, and evaluation for different topic modeling techniques.
The API grant common interfaces and command for accessing the different models, make easier to compare them.

A demo is available at http://hyperted.eurecom.fr/topic.

## Models and Dataset

In this repository, we provide:

* Code to perform training, inference, and evaluation for 4 Topic Modeling packages:
  * LDA from the [Mallet](http://mallet.cs.umass.edu/) package.
  * [LFTM](https://github.com/datquocnguyen/LFTM) (LF-LDA) - [paper](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158).
  * [Doc2Topic](https://github.com/sronnqvist/doc2topic), NTM in the code
  * [GSDMM](https://github.com/rwalk/gsdmm) - [paper](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf)
  * TFIDF
* Pre-trained models in the **models/** directory. These are automatically used in the inference and evaluation API calls. The models have been trained using [these parameters](./params.md). **NOTE: Newly trained models are stored there, but replace the old ones.**
* Data files containing all subtitles available on TED as well as the tags of TED videos.

## How to use the API

The API is intended to be used in a [Docker](https://www.docker.com/) environment.

#### Install dependencies

You should install 2 dependencies:
- [mallet 2.0.8](http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz) to be placed in `app\builtin`
- [glove.6B.50d.txt](http://nlp.stanford.edu/data/glove.6B.zip) to be placed in `app\builtin\glove`

Under UNIX, you can use the **download_dep.sh** script.

    sh download_dep.sh


#### Production

    docker-compose -f docker-compose.yml up

This version will copy the TED data need for training and some dependencies like Glove word vectors for LFTM.

#### Dev Mod

    docker-compose -f docker-compose-dev.yml up

This command will build the image from the Dockerfile in **app/deployment/** and start the container.

The container uses **mounted volumes** so that you can easily update the code, the model files, and the data files.

#### Manual Docker installation

    docker build -t hyperted/topic .
    docker run -p 27020:5000  -d -v /home/semantic/TopicModelAPI/ted-talk-topic-extraction/models:/models -v /home/semantic/TopicModelAPI/ted-talk-topic-extraction/data:/data --name hyperted_topic hyperted/topic

    # Uninstall
    docker stop hyperted_topic
    docker rm hyperted_topic
    docker rmi hyperted/topic
