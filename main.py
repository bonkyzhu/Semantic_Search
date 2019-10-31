from match import *
from queue import PriorityQueue
from multiprocessing.pool import ThreadPool

files = gci('./')
idf, bm25, tfidf = save_IDF_and_BM25_and_TFIDF('./')
text = seg(input("输入你需要查询的：\n"))
q = PriorityQueue()

def score(file, method='BM25'):
  if file not in bm25.keys() or file not in tfidf.keys():
    return
  if method == 'BM25':
    tmp = bm25[file]
  if method == 'TFIDF':
    tmp = tfidf[file]
  tmp_score = 0
  for token in text:
    if token in tmp.keys():
      tmp_score += tmp[token]
  q.put((-tmp_score, file))

pool = ThreadPool(12)
pool.map(score, files)
pool.close()

print('-'*80 + '\n找到以下匹配的文章')
for i in range(5):
  print(q.get()[1])
print('\n找到以下匹配的文章')