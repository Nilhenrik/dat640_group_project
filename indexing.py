import pyterrier as pt
import pandas as pd
if not pt.started():
  pt.init() 


print("Reading file")
df = pd.read_csv('./data/collection.tsv', sep='\t', names=['docno', 'text'],dtype=str, encoding='utf-8')
print("Dataframe read")
df.dropna(inplace=True)
res_dict = df.to_dict()
# Indexing using PyTerrier
index_loc = "./terrier_index"
indexer = pt.IterDictIndexer(index_loc, overwrite=True, verbose=True,tokeniser="UTFTokeniser")
print("Indexer made")
index = indexer.index(df.to_dict(orient="records"))
#index = indexer.index(res_dict["text"], res_dict["docno"])
print("Indexer indexed")
print("Done")
