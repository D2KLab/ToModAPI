FROM ubuntu

RUN mkdir /app
WORKDIR /app

RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install default-jre -y


COPY ./deployment/prod/req.txt req.txt
RUN pip3 install -r req.txt

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet");'

ADD ./ ./


CMD ["uwsgi", "project.ini"]