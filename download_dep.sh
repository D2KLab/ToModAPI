#!/bin/bash
MAIN_DIR=$(dirname "$0")
wget -P $(echo $MAIN_DIR) http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
tar -xf $(echo $MAIN_DIR)/mallet-2.0.8.tar.gz -C $(echo $MAIN_DIR)/app/builtin
rm -f $(echo $MAIN_DIR)/mallet-2.0.8.tar.gz
rm -f $(echo $MAIN_DIR)/app/builtin/._*

wget -P $(echo $MAIN_DIR) http://nlp.stanford.edu/data/glove.6B.zip
tar -xvf $(echo $MAIN_DIR)/glove.6B.zip -C $(echo $MAIN_DIR)/app/builtin/glove
rm -f $(echo $MAIN_DIR)/glove.6B.zip


