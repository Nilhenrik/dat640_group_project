import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pyterrier as pt

import pandas as pd
import pyterrier_doc2query
if not pt.started():
  pt.init() 


print("Reading file")
df = pd.read_csv('./data/collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
print("Dataframe read")
df.dropna(inplace=True)

# Indexing setup
index_loc = "./index"

# We chain the doc2query transformation with the IterDictIndexer for indexing
indexer =  pt.IterDictIndexer(index_loc, meta={"docno": 20, "text": 4096}, overwrite=True,tokeniser="UTFTokeniser")
print("Indexer made")

# Trigger the indexing process
index = indexer.index(df.to_dict(orient="records"))
print("Indexer indexed")
print("Done")
