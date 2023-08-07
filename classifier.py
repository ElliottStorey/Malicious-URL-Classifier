from urllib.parse import urlparse
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec

try:
    model = Word2Vec.load('word2vec_model.bin')
except FileNotFoundError:
    raise FileNotFoundError('Cannot find Word2Vec model. Run word2vec.py to generate the model.')

def tokenizer(url):
    if not urlparse(url).scheme:
        return urlparse(f'//{url}')
    else:
        return urlparse(url)

def vectorizer(url):
    tokens = tokenizer(url)
    vectors = [model.wv.get_vector(token) for token in tokens if token in model.wv]
    vector = torch.mean(torch.tensor(np.array(vectors, dtype=np.float32)), dim=0)
    return vector

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(100, 75),
            torch.nn.Sigmoid(),
            torch.nn.Linear(75, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

classifier = Net()

try:
    classifier.load_state_dict(torch.load('classifier.pth'))
except:
    pass

def predict(url):
    url_vector = vectorizer(url)
    with torch.no_grad():
        output = classifier(url_vector)
    output = float(torch.squeeze(output))
    return output