import re
import nltk

init_done = False


def is_list_of_strings(lst):
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def input_to_list_string(data, preprocessing=False):
    if type(data) == str:
        with open(data, "r", encoding='utf-8') as datafile:
            text = [line.rstrip() for line in datafile if line]
    elif is_list_of_strings(data):
        text = data
    else:
        raise ValueError('data should be a path or a list of strings')
    if preprocessing:
        text = [preprocess(doc) for doc in text]
    return text


def _init():
    global init_done
    nltk.download('stopwords')
    nltk.download('wordnet')
    init_done = True


def preprocess(text, strip_brackets=False):
    """ Preprocessing function for text, consisting of:
    - Removing numbers, which, in general, do not contribute to the broad semantics;
    - Removing the punctuation and lower-casing the text;
    - Removing the standard English stop words;
    - Lemmatisation using Wordnet, in order to deal with inflected forms as they are a single semantic item;
    - Ignoring words with 2 letters or less. In facts, they are mainly residuals from removing punctuation
            e.g. stripping punctuation from _people’s_ produces _people_ and _s_

    :param str text: the text to be preprocessed
    :param bool strip_brackets: If True, the content inside brackets is excluded
    """
    global init_done

    if not init_done:
        _init()

    if strip_brackets:
        text = re.sub(r'\((.*?)\)', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower())
    text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]

    lem = nltk.stem.WordNetLemmatizer()

    text = [lem.lemmatize(w) for w in text]
    text = [w for w in text if len(w) >= 3]
    text = ' '.join(text)
    return text
