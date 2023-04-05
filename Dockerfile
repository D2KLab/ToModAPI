# Previously the following: use openjdk image, then copy over root directory from python img
# Potentially problematic; might overwrite other necessary files
# FROM openjdk:slim
# COPY --from=python:3.6.5 / /


# Uses python as base image
FROM python:3.6.5

# then installs java as well
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

#COPY download_dep.sh .
#RUN download_dep.sh

RUN mkdir tomodapi
ADD tomodapi tomodapi
ADD server.py .
ADD project.ini .

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("omw-1.4");'

CMD ["uwsgi", "project.ini"]