FROM ubuntu

RUN mkdir /app
WORKDIR /app

RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install default-jre -y


COPY req.txt req.txt
RUN pip3 install -r req.txt

CMD export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && flask run --host=0.0.0.0 --port=5000