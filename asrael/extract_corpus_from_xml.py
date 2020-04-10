#!/usr/bin/python
# coding: utf-8

import os
import re
import nltk
import argparse
import fnmatch
from tqdm import tqdm
from xml.dom import minidom

lem = nltk.stem.WordNetLemmatizer()

output_path = ""


def preprocess(text):
    text = re.sub(r'\((.*?)\)', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower())
    text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]
    text = [w for w in text if len(w) >= 3]
    text = [lem.lemmatize(w) for w in text]
    text = ' '.join(text)
    return text


def get_text(nodelist):
    # Iterate all Nodes aggregate TEXT_NODE
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
        else:
            # Recursive
            rc.append(get_text(node.childNodes))
    return " ".join(rc)


def parse_worker(xml_file_path):
    global output_path

    basename = os.path.splitext(os.path.basename(xml_file_path))[0]
    txt_file_path = os.path.join(output_path, "{}.txt".format(basename)).encode("utf8")
    if os.path.isfile(txt_file_path):
        # print("skipping {} because {} already exists".format(xml_file_path, txt_file_path))
        return

    # print("processing {}".format(xml_file_path))

    doc = minidom.parse(xml_file_path)
    headline_nodes = doc.getElementsByTagName("HeadLine")
    if headline_nodes and headline_nodes[0].firstChild:
        if headline_nodes[0].firstChild.nodeValue:
            headline = headline_nodes[0].firstChild.nodeValue
        else:
            headline = headline_nodes[0].firstChild.firstChild.nodeValue
    else:
        headline = ""

    # extract subject
    subject_nodes = doc.getElementsByTagName("SubjectCode")
    subj1 = []
    subj2 = []
    subj3 = []

    for node in subject_nodes:
        for child in node.childNodes:
            tag = child.nodeName
            if tag == 'SubjectMatter':
                subj2.append(child.attributes['FormalName'].value)
            elif tag == 'Subject':
                subj1.append(child.attributes['FormalName'].value)
            elif tag == 'SubjectDetail':
                subj3.append(child.attributes['FormalName'].value)

    subj1 = set(subj1)
    subj2 = set(subj2)
    subj3 = set(subj3)

    # extract text
    text_nodes = doc.getElementsByTagName("DataContent")
    if len(text_nodes) > 0:
        corpus = get_text(text_nodes[0].childNodes).strip()
    else:
        corpus = get_text(doc.getElementsByTagName("Content")[0].childNodes).strip()

    if headline not in corpus:
        corpus = headline + ".\n\n" + corpus.strip()

    with open(txt_file_path, "w") as out:
        out.write(' '.join(subj1))
        out.write('\n')
        out.write(' '.join(subj2))
        out.write('\n')
        out.write(' '.join(subj3))
        out.write('\n\n')
        out.write(corpus)


def main():
    global output_path

    # -- process arguments
    parser = argparse.ArgumentParser(
        description="Parses AFP newswires XML files and generates corpus TXT files")
    parser.add_argument('-i', "--input", type=str, default='tlp.limsi.fr',
                        help="Path to folder containing the XML files to parse")
    parser.add_argument('-o', "--output", type=str, default='text',
                        help="Paths to the folder where the TXT outputs will be stored")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = args.output

    corpus_files_paths = []
    for root, dirnames, filenames in sorted(os.walk(args.input)):
        for filename in fnmatch.filter(filenames, "*.xml"):
            corpus_files_paths.append(os.path.join(root, filename))

    nb_proc = 16

    if len(corpus_files_paths) < nb_proc:
        nb_proc = len(corpus_files_paths)
    # '''
    # Monothread version
    for corpus_file_path in tqdm(corpus_files_paths):
        parse_worker(corpus_file_path)
    # '''

    '''
    p = Pool(processes=nb_proc)
    for corpus_file_path in corpus_files_paths:
        # We could use apply_async for even better parallelization handling, but we are doing a lot of writing files, which frees out the worker even if the writing is not completely done, resulting in pursuing execution while all files are not written (so can't be accessed, so errors)
        p.apply_async(parseWorker, (corpus_file_path,))
    p.close()
    p.join()
    '''

    # create unique corpus in the end
    corpus = []
    subj = []
    for filename in tqdm(sorted(os.listdir(args.output))):
        with open(os.path.join(args.output, filename), "r") as f:
            lines = [l.strip() for l in f.readlines()]

        subj.append(','.join(lines[0:3]))
        corpus.append('\\n'.join(lines[4:]))

    with open('./afp.txt', "w") as f:
        for l in tqdm(corpus):
            f.write(preprocess(l))
            f.write('\n')
    with open('./afp_labels.txt', "w") as f:
        for l in tqdm(subj):
            f.write(l)
            f.write('\n')


if __name__ == '__main__':
    main()
