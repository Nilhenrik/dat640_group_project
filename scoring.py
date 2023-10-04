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
queries_file = "./data/queries_train.csv"
index_loc = "./terrier_index"
qrels_file = "./data/qrels_train.txt"

index = pt.IndexFactory.of(index_loc)
bm25 = pt.BatchRetrieve(index, wmodel="BM25",controls={"bm25.k1": 1.5, "bm25.b": 0.5})
queries = pd.read_csv(queries_file)
qrels = pt.io.read_qrels(qrels_file)
queries['query'] = queries['query'].apply(preprocess)

results = bm25.transform(queries[['qid', 'query']])
pt.io.write_results(results, "./data/bm25score.txt")
#eval_metrics = ["map", "ndcg", "P_5", "P_10"]
#evaluator = pt.Utils.evaluate(results, qrels, metrics=eval_metrics)
