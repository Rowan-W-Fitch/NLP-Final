import sys
import re
import os
import numpy as np
import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

folder = sys.argv[1]
file = open(os.path.join(folder,"Hawaii.txt"), encoding = 'utf-8')

sentences = []
for sentence in nltk.sent_tokenize(file.read()):
    sentences.append(sentence)

big_file = open('glove/glove.6B.100d.txt', encoding = 'utf-8')
word_embs = {}
for line in big_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    word_embs[word] = coefs
big_file.close()

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]
stp_wrds = nltk.corpus.stopwords.words('english')

def clear_stp(sentence):
    new_sent = " ".join([w for w in sentence if w not in stp_wrds])
    return new_sent

clean_sentences = [clear_stp(r.split()) for r in clean_sentences]
vctrs = []
for c in clean_sentences:
    if len(c) == 0:
        v = np.zeros((100,))
    else:
        v = sum([word_embs.get(w, np.zeros((100,))) for w in c.split()])/(len(c.split()) + 0.001)
    vctrs.append(v)

matrix = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i!=j:
            matrix[i][j] = cosine_similarity(vctrs[i].reshape(1,100), vctrs[j].reshape(1,100))[0,0]

graph = nx.from_numpy_array(matrix)
scores = nx.pagerank(graph)
ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
for i in range(10):
    print(ranked[i][1])
