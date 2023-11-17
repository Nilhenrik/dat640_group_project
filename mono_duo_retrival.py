import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker,DuoT5ReRanker
import pandas as pd
import re

# Function to remove special characters from the query
def preprocess(doc: str) -> str:
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
    ])

# Loads castorini/monot5-base-msmarco by default
monoT5 = MonoT5ReRanker(verbose=True, batch_size=32) 
duoT5 = DuoT5ReRanker(verbose=True, batch_size=32)

# Start pyterrier if it hasnt started
if not pt.started():
  pt.init()

# Queries and index path
queries_file = "./data/queries_train.csv"
index_loc = "./index"

# Load the index and pipeline
index = pt.IndexFactory.of(index_loc)
bm25 = pt.BatchRetrieve(index, wmodel="BM25",controls={"bm25.k1": 1.5, "bm25.b": 0.5})
mono_pipeline = bm25 >> pt.text.get_text(index, "text") >> monoT5
duo_pipeline = mono_pipeline % 50 >> duoT5

# Read the queries
queries = pd.read_csv(queries_file)
queries['query'] = queries['query'].apply(preprocess)

# Run the pipeline and save the results
results = duo_pipeline.transform(queries[['qid', 'query']])
pt.io.write_results(results, "./data/monoduo.txt")
