FROM openjdk:slim
COPY --from=python:3.6.5 / /

RUN mkdir /app
WORKDIR /app

COPY app/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY patch_swagger.sh /app
RUN /app/patch_swagger.sh

ADD /app .

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet");'

CMD ["uwsgi", "project.ini"]