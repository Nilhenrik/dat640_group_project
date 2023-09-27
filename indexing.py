import re
from typing import List
from sqlitedict import SqliteDict
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
from scorer import ScorerBM25, DocumentCollection


def preprocess(doc: str) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]
class InvertedIndex(SqliteDict):
    def __init__(
        self,
        filename: str = "inverted_index.sqlite",
        new: bool = False,
    ) -> None:
        super().__init__(filename, flag="n" if new else "c")
        self.index ={} if new else self
        self.collection ={} if new else self
        
    def add_posting(self,term:str,doc_id:int)->None:
        if term not in self.index:
            self.index[term] = {}
        if doc_id in self.index[term]:
            self.index[term][doc_id]+=1
        else:
            self.index[term][doc_id]=1
            
    def __exit__(self, *exc_info):
        if self.flag == "n":
            self.update(self.index)
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)
            

if __name__ == "__main__":
    with open('collection.tsv', 'r') as file:
        with InvertedIndex('inverted_index.sqlite', new=True) as index:
            for line in file:
                fields = line.strip().split('\t')
                terms = preprocess(fields[1])
                doc_id = int(fields[0])
                index.collection[doc_id] = terms
                for term in terms:
                    if term is None:continue
                    index.add_posting(term, doc_id)
            scorer_bm25 = ScorerBM25(DocumentCollection(index.collection), index.index)
            with open('queries_train.csv','r') as f:
                next(f)
                index = 0
                for line in f:
                    if index==1: break
                    columns = line.split(',')
                    turn_indentifier = columns[0]
                    placeholder = "Q0"
                    test = scorer_bm25.score_collection(preprocess(columns[1]))
                    top_results = scorer_bm25.get_top_n_results(1000)
                    for i, res in enumerate(top_results):
                        doc_id = res[0]
                        doc_score = res[1]
                        print(f"{turn_indentifier} {placeholder} {doc_id} {i} {doc_score} BM25")
                    index+=1