from tqdm import tqdm
from topic_modeling.utils.corpus import preprocess

with open('data/20ng.txt') as f:
    corpus = f.readlines()

text = [preprocess(x) for x in tqdm(corpus)]

with open('data/20ng.txt', 'w') as f:
    f.writelines(text)
