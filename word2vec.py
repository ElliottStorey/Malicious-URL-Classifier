from urllib.parse import urlparse
import pandas as pd
from gensim.models import Word2Vec

data = pd.read_csv('data.csv')

def tokenizer(url):
    if not urlparse(url).scheme:
        return urlparse(f'//{url}')
    else:
        return urlparse(url)

tokens = data['url'].apply(tokenizer)

model = Word2Vec(sentences=tokens, min_count=1)

model.save('word2vec_model.bin')