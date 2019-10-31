'''
计算BM25
公式：simlarity = IDF * ((k + 1) * tf) / (k * (1.0 - b + b * (|d|/avgDl)) + tf)
     TF = sqrt(词在一篇文章的数目/文章的总词数)
     IDF = log(总共的文章数/含有这个词的文章数目)
'''
import pkuseg
from math import sqrt
from os.path import exists
import numpy as np
import string 
import pickle
from multiprocessing.pool import ThreadPool

PATH = './'

def seg(article):
  seg = pkuseg.pkuseg()
  return seg.cut(article)

def to_TF(article):
  '''
  计算文章/句子所有词的TF值，返回一个字典
  :param article: str
  :return article_TF: dict
  '''
  text = seg(article)
  length = len(text)
  article_TF = {}
  for token in text:
    if token not in article_TF.keys():
      article_TF[token] = 1
    else:
      article_TF[token] += 1
  for key in article_TF.keys():
    article_TF[key] /= length
    article_TF[key] = round(sqrt(article_TF[key]), 4)
  return article_TF

def to_IDF(corpus):
  '''
  计算给定的所有文章中， 得到一个vocab，vocab有所有词以及他们的词频
  :param corpus: list, 其中每一个元素为str
  :return corpus_IDF: 
  '''
  corpus_IDF = {}
  for article in corpus:
    word_set = set(seg(article))
    for word in word_set:
      if word not in corpus_IDF.keys():
        corpus_IDF[word] = 1
      else:
        corpus_IDF[word] += 1
  return corpus_IDF

def BM25(article, avgDl, corpus_IDF, k=2, b=0.75):
  '''
  :param k, b
         d: 当前文章的长度
         avgdl当前文章的长度
  '''
  text = set(seg(article))
  d = len(text)
  similarity = {}
  article_TF = to_TF(article)
  for token in text:
    idf = corpus_IDF[token]
    tf = article_TF[token]
    similarity[token] = idf * ((k + 1) * tf) / (k * (1.0 - b + b * (d/avgDl)) + tf)
    similarity[token] = round(similarity[token], 4)
  return similarity

def TFIDF(article, corpus_IDF):
  text = set(seg(article))
  d = len(text)
  similarity = {}
  article_TF = to_TF(article)
  for token in text:
    idf = corpus_IDF[token]
    tf = article_TF[token]
    similarity[token] = idf * tf
    similarity[token] = round(similarity[token], 4)
  return similarity

def gci(filepath, print_file=False):
  #遍历filepath下所有文件，包括子目录
  import os
  lists = []
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      lists += gci(fi_d, print_file)                  
    elif fi_d.endswith('.md'):
      lists.append(fi_d)
      if print_file:
        print(fi_d)
  return lists

def save_IDF_and_BM25_and_TFIDF(corpus_path):
  corpus = []
  files = gci(corpus_path)
  avgDl = 0

  def read2corpus(file):
    text = open(file).read()
    corpus.append(text)

  pool = ThreadPool(12)
  pool.map(read2corpus, files)
  pool.close()

  tmp = np.array([len(article) for article in corpus])
  avgDl = np.sum(tmp) / len(files)

  if not exists('idf.dict'):
    idf = to_IDF(corpus)
    with open('idf.dict', "wb") as f:
      pickle.dump(idf, f)
  else:
    idf = pickle.load(open('idf.dict', 'rb'), encoding='bytes')

  if not exists('BM25.dict'):
    bm25 = {}
    for file in files:
      article = open(file).read()
      bm25[file] = BM25(article, avgDl, idf)
    with open('BM25.dict', "wb") as f:
      pickle.dump(bm25, f)
  else:
    bm25 = pickle.load(open('BM25.dict', 'rb'), encoding='bytes')

  if not exists('TFIDF.dict'):
    tfidf = {}
    for file in files:
      article = open(file).read()
      tfidf[file] = TFIDF(article, idf)
    with open('TFIDF.dict', "wb") as f:
      pickle.dump(tfidf, f)
  else:
    tfidf = pickle.load(open('TFIDF.dict', 'rb'), encoding='bytes')
  
  return idf, bm25, tfidf

if __name__ == '__main__':
  save_IDF_and_BM25_and_TFIDF(PATH)
