#!/bin/bash
MAIN_DIR=$(dirname "$0")
wget -P $(echo $MAIN_DIR) http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
tar -xf $(echo $MAIN_DIR)/mallet-2.0.8.tar.gz -C $(echo $MAIN_DIR)/topic_modeling
rm -f $(echo $MAIN_DIR)/mallet-2.0.8.tar.gz
rm -f $(echo $MAIN_DIR)/app/builtin/._*

wget -P $(echo $MAIN_DIR) http://nlp.stanford.edu/data/glove.6B.zip
unzip $(echo $MAIN_DIR)/glove.6B.zip -d $(echo $MAIN_DIR)/topic_modeling/glove
rm -f $(echo $MAIN_DIR)/glove.6B.zip


