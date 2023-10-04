import pyterrier as pt
import pandas as pd
from typing import List
if not pt.started():
  pt.init() 


print("Reading file")
df = pd.read_csv('collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
print("dataframe read")
df.dropna(inplace=True)
res_dict = df.to_dict()
# Indexing using PyTerrier
index_loc = "./terrier_index"
indexer = pt.IterDictIndexer(index_loc, overwrite=True, verbose=True)
print("indexer made")
index = indexer.index(df.to_dict(orient="records"))
#index = indexer.index(res_dict["text"], res_dict["docno"])
print("indexer indexed")
print("Done")
