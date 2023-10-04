import pyterrier as pt
if not pt.started():
  pt.init() 

def msmarco_generate():
  dataset = pt.get_dataset("collection.csv")
  with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
      for l in corpusfile:
          docno, passage = l.split("\t")
          yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./passage_index", overwrite=True)
indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096})

""" def msmarco_generate():
    dataset = pt.get_dataset("trec-deep-learning-passages")
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./passage_index")
indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096}) """