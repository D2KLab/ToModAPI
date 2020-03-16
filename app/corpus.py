import re

import nltk
import numpy as np
import requests

# Download stop words and wordnet for data cleaning and prepocessing
nltk.download('stopwords')
nltk.download('wordnet')
lem = nltk.stem.WordNetLemmatizer()


# Extract video subtitles from TED URL
def retrieve_prepare_subtitles(url):
    # Extract video title
    title = url.split('/')[-1]
    # Build metadata URL
    metaURL = 'https://api.ted.com/v1/talks/{}.json?api-key=uzdyad5pnc2mv2dd8r8vd65c'
    # Get Metadata
    r = requests.get(url=metaURL.format(title))
    data = r.json()
    # Verify the talk exists
    if 'talk' in data:
        # Get video id
        id = data['talk']['id']
        # Build subtitles url
        subURL = 'https://api.ted.com/v1/talks/{}/subtitles.json?api-key=uzdyad5pnc2mv2dd8r8vd65c'
        # Get video subtitles
        r = requests.get(url=subURL.format(id))
        data = r.json()
        # Verify the subtitles exists
        if 'error' in data:
            return 'not found'
        transcript = ''
        # Preprocess every section of the subtitles
        for dict in np.arange(len(data) - 1):
            text = data[str(dict)]['caption']['content']
            text = re.sub(r'\((.*?)\)', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower())
            text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]
            text = [nltk.corpus.wordnet.morphy(w) if nltk.corpus.wordnet.morphy(w) else w for w in text]
            text = ' '.join(text)
            transcript = ' '.join([transcript, text])
        # Return subtitles
        return transcript
    return 'not found'


# Preprosses subtitles
def prepare_subtitles(text):
    text = re.sub(r'\((.*?)\)', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower())
    text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]
    text = [lem.lemmatize(w) for w in text]
    text = [w for w in text if len(w) >= 3]
    text = ' '.join(text)
    return text


# Extract video tags from TED URL
def retrieve_prepare_tags(url):
    # Extract video title
    title = url.split('/')[-1]
    # Build metadata URL
    metaURL = 'https://api.ted.com/v1/talks/{}.json?api-key=uzdyad5pnc2mv2dd8r8vd65c'
    r = requests.get(url=metaURL.format(title))
    data = r.json()
    # Retrieve video tags
    tags = ','.join([tag['tag'] for tag in data['talk']['tags']])
    # Return tags
    return tags
