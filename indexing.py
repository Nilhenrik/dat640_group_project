import re
from typing import List
import pandas as pd
from sqlitedict import SqliteDict
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

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
                for term in preprocess(fields[1]):
                    if term is None:continue
                    index.add_posting(term, fields[0])