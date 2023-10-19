import pyterrier as pt
import pandas as pd
import pyterrier_doc2query
if not pt.started():
  pt.init() 


print("Reading file")
df = pd.read_csv('./data/collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
print("Dataframe read")
df.dropna(inplace=True)
doc2query = pyterrier_doc2query.Doc2Query(append=True) # append generated queries to the original document text

# Indexing setup
index_loc = "./terrier_index"

# We chain the doc2query transformation with the IterDictIndexer for indexing
indexer = doc2query >> pt.IterDictIndexer(index_loc, overwrite=True, verbose=True, tokeniser="UTFTokeniser")
print("Indexer made")

# Trigger the indexing process
index = indexer.index(df.to_dict(orient="records"))
print("Indexer indexed")
print("Done")
