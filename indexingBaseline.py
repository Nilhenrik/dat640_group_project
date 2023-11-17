import pyterrier as pt
import pandas as pd
if not pt.started():
  pt.init() 

# Index path
index_loc = "./index"

#Read the collection file and drop empty lines
df = pd.read_csv('./data/collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
df.dropna(inplace=True)

# Dfine the IterDictIndexer
indexer = pt.IterDictIndexer(index_loc, meta={"docno": 20, "text": 4096}, overwrite=True, verbose=True, tokeniser="UTFTokeniser")

# Trigger the indexing process
index = indexer.index(df.to_dict(orient="records"))