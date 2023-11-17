import pyterrier as pt
import pandas as pd
import re


# Function to remove special characters from the query
def preprocess(doc: str) -> str:
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
    ])

# Start pyterrier if it hasnt startet yet
if not pt.started():
  pt.init()

# Queries and index path
queries_file = "./data/queries_train.csv"
index_loc = "./index"

# Load the index and bm25 
index = pt.IndexFactory.of(index_loc)
bm25 = pt.BatchRetrieve(index, wmodel="BM25",controls={"bm25.k1": 1.5, "bm25.b": 0.5})

# Read and preprocess the queries
queries = pd.read_csv(queries_file)
queries['query'] = queries['query'].apply(preprocess)

# Retrieve the documents and save the results
results = bm25.transform(queries[['qid', 'query']])
pt.io.write_results(results, "./data/bm25score.txt")