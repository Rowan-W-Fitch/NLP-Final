import sys
import re
import os
import numpy as np
import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

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
    scores = nx.pagerank(graph)
    ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
    #get summary
    summary = " ".join([ranked[i][1] for i in range(10)])
    #write summary to a file
    cg_summ = open(os.path.join(CG_Summary_folder, f), 'w', encoding = 'utf-8')
    cg_summ.write(summary)
    cg_summ.close()
    file.close()
print('summaries written') #status update

#little function that preps the cg summary for nltk bleu score
def bleu_prep_cg(doc):
    ref = []
    for sentence in nltk.sent_tokenize(doc):
        ref.append(nltk.word_tokenize(sentence))
    return ref

#function preps human summary for nltk bleu score
def bleu_prep_human(doc):
    hyp = []
    for sentence in nltk.sent_tokenize(doc):
        hyp.append(nltk.word_tokenize(sentence))
    return hyp

#get ROGUE score for each of the summaries
rs = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer = True)
for order in orders:
    #open files
    human_summary = os.path.join(Human_Summary_folder, order)
    cg_summary = os.path.join(CG_Summary_folder, order)
    #use rouge package
    r_scores = rs.score(open(cg_summary, encoding = 'utf-8').read(), open(human_summary, encoding = 'utf-8').read())
    b_score = corpus_bleu(bleu_prep_cg(open(cg_summary, encoding = 'utf-8').read()), bleu_prep_human(open(human_summary, encoding = 'utf-8').read()))
    #print scores
    print('ROUGE scores: ', r_scores)
    print('BLEU score: ', b_score)
    #put this in here in case the user wants to see what the cg summary looks like
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(" ")
    human_summary.close()
    cg_summary.close()
