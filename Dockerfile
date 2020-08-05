FROM openjdk:slim
COPY --from=python:3.6.5 / /

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

#COPY download_dep.sh .
#RUN download_dep.sh

RUN mkdir topic_model
ADD topic_model topic_model
ADD server.py .
ADD project.ini .

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet");'

CMD ["uwsgi", "project.ini"]