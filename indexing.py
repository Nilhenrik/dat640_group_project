import pyterrier as pt
import pandas as pd
import pyterrier_doc2query
if not pt.started():
  pt.init() 

# Index path
index_loc = "./index5"

#Read the collection file and drop empty lines
df = pd.read_csv('./data/collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
df.dropna(inplace=True)

# Chain the doc2query transformation with the IterDictIndexer
doc2query = pyterrier_doc2query.Doc2Query(append=True, fast_tokenizer=True, num_samples=5) # append generated queries to the original document text
indexer = doc2query >> pt.IterDictIndexer(index_loc, meta={"docno": 20, "text": 4096}, overwrite=True, verbose=True, tokeniser="UTFTokeniser")

# Trigger the indexing process
index = indexer.index(df.to_dict(orient="records"))