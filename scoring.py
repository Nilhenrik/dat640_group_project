import pyterrier as pt
import pandas as pd
import re
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def preprocess(doc: str) -> str:
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ])

if not pt.started():
  pt.init() 
queries_file = "queries_train.csv"
index_loc = "./terrier_index"
qrels_file = "qrels_train.txt"

index = pt.IndexFactory.of(index_loc)
bm25 = pt.BatchRetrieve(index, wmodel="BM25",controls={"bm25.k1": 1, "bm25.b": 0.5,"bm25.k_3":10})
queries = pd.read_csv(queries_file)
qrels = pt.io.read_qrels(qrels_file)
queries['query'] = queries['query'].apply(preprocess)

results = bm25.transform(queries[['qid', 'query']])

pt.io.write_results(results, "bm25score.txt")
