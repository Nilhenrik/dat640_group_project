import pyterrier as pt
import pandas as pd
import re
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
from tqdm import tqdm

def preprocess(doc: str) -> str:
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ])

if not pt.started():
  pt.init() 
test_topics_file = "queries_train.csv"
index_loc = "./terrier_index"
test_qrels_file = "qrels_train.txt"

index = pt.IndexFactory.of(index_loc)
bm25 = pt.BatchRetrieve(index, wmodel="BM25",controls={"bm25.k1": 1, "bm25.b": 0.5,"bm25.k_3":10})
queries = pd.read_csv(test_topics_file)
qrels = pt.io.read_qrels(test_qrels_file)
queries['query'] = queries['query'].apply(preprocess)

test_results = bm25.transform(queries[['qid', 'query']])

pt.io.write_results(test_results, "bm25score.txt")
eval_metrics = ["map", "ndcg", "P_5", "P_10"]
evaluator = pt.Utils.evaluate(test_results, qrels, metrics=eval_metrics)
print(evaluator)