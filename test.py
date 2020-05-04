import sys
import re
import os
import numpy as np
import nltk
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from queue import PriorityQueue

SIP_folder = sys.argv[1] #name of folder holding Shelter in place orders as txt files, SIP = Shelter In Place
CG_Summary_folder = sys.argv[2] #name of folder holding computer generated summaries
Human_Summary_folder = sys.argv[3] #name of folder holding human written summaries

#create word embeddings with glove corpora (6 billion words)
big_file = open('glove/glove.6B.100d.txt', encoding = 'utf-8')
word_embs = {}
for line in big_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    word_embs[word] = coefs
big_file.close()
print("embedding done") #status update

#simple function for removing stop words
def clear_stp(sentence):
    new_sent = " ".join([w for w in sentence if w not in stp_wrds])
    return new_sent

#list of SIP orders
orders = []
#go thru each txt file and get the summary, then writes summary to a file
for f in os.listdir(SIP_folder):
    orders.append(f)
    file = open(os.path.join(SIP_folder,f), encoding = 'utf-8')
    #tokenize sentences
    sentences = []
    for sentence in nltk.sent_tokenize(file.read()):
        sentences.append(sentence)
    #get rid of non alphanumeric chars, lowercase all txts, and remove stop words
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    stp_wrds = nltk.corpus.stopwords.words('english')
    clean_sentences = [clear_stp(r.split()) for r in clean_sentences]
    #represent words in txt as vectors using embeddings from top of file
    vctrs = []
    for c in clean_sentences:
        if len(c) == 0:
            v = np.zeros((100,))
        else:
            v = sum([word_embs.get(w, np.zeros((100,))) for w in c.split()])/(len(c.split()) + 0.001)
        vctrs.append(v)
    #represent txt as a sparse matrix
    matrix = np.zeros([len(sentences), len(sentences)])
    #fill matrix w/ vectors
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i!=j:
                matrix[i][j] = cosine_similarity(vctrs[i].reshape(1,100), vctrs[j].reshape(1,100))[0,0]
    #represent matrix as a graph, and perform pagerank algorithm
    graph = nx.from_numpy_array(matrix)
    scores = nx.pagerank_numpy(graph)
    ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
    #get summary
    summary = " ".join([ranked[i][1] for i in range(10)])
    #write summary to a file
    cg_summ = open(os.path.join(CG_Summary_folder, f), 'w', encoding = 'utf-8')
    cg_summ.write(summary)
    cg_summ.close()
    file.close()
print('summaries written') #status update

#get metrics, and sentences common among all summaries
avg_pr1, avg_prL= 0, 0
avg_rec1, avg_recL = 0, 0
avg_fm1, avg_fmL  = 0, 0
rs = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer = True)
sentences = []
len = 0
max_pr1, max_rec1, max_f1 = 0, 0, 0
max_prL, max_recL, max_fL = 0, 0, 0
for order in orders:
    #open files
    len +=1
    human_summary = open(os.path.join(Human_Summary_folder, order), encoding = 'utf-8')
    cg_summary = open(os.path.join(CG_Summary_folder, order), encoding = 'utf-8')
    #get txt
    cg_txt = cg_summary.read()
    human_txt = human_summary.read()
    #use rouge package
    r_scores = rs.score(cg_txt, human_txt)
    #add sentences to list(for simmilar sentences among all docs)
    for sentence in sent_tokenize(cg_txt):
        sentences.append(sentence)
    #avg metrics for 1-gram
    avg_pr1 += r_scores['rouge1'][0]
    max_pr1 = max(r_scores['rouge1'][0], max_pr1)
    avg_rec1 += r_scores['rouge1'][1]
    max_rec1 = max(r_scores['rouge1'][1], max_rec1)
    avg_fm1 += r_scores['rouge1'][2]
    max_f1 = max(r_scores['rouge1'][2], max_f1)
    #avg metrics for L
    avg_prL += r_scores['rougeL'][0]
    max_prL = max(r_scores['rougeL'][0], max_prL)
    avg_recL += r_scores['rougeL'][1]
    max_recL = max(r_scores['rougeL'][1], max_recL)
    avg_fmL += r_scores['rougeL'][2]
    max_fL = max(r_scores['rougeL'][2], max_fL)
    human_summary.close()
    cg_summary.close()

avg_pr1/=len
avg_prL/=len
avg_rec1/=len
avg_recL/=len
avg_fm1/=len
avg_fmL/=len
print("avg precisions: ", "r1 -> ", avg_pr1, " rL -> ", avg_prL)
print("avg recalls: ", "r1 -> ", avg_rec1, " rL -> ", avg_recL)
print("avg fmeasures: ", "r1 -> ", avg_fm1, " rL -> ", avg_fmL)
print("max pr1 ", max_pr1, " max rec1 ", max_rec1, " max_f1 ", max_f1)
print("max prL ", max_prL, " max recL ", max_recL, " max_fL ", max_fL)
#after metrics, get simmilar sentences among all docs
#create dict of all words
words = [[w.lower() for w in word_tokenize(sentence)]
            for sentence in sentences]
dictionary = gensim.corpora.Dictionary(words)
#create corpus
corpus = [dictionary.doc2bow(word) for word in words]
#score sentences on TFIDF
tfidf = gensim.models.TfidfModel(corpus)
#convert each sentence into sum of TFIDF score of each word
#get top 5 scores w/ priority queue
Q = PriorityQueue()
for doc, sent in zip(tfidf[corpus], sentences):
    score = 0
    for id, freq in doc:
        score += freq
    Q.put((score, sent))

for i in range(20):
    print(Q.get()[1])
