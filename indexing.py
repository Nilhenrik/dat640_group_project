import re
from typing import List, Dict, Tuple
from sqlitedict import SqliteDict
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
from tqdm import tqdm

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
        self.index:Dict[str, Dict[Tuple[int,int]]] ={} if new else self
        self.collection ={} if new else self
        
    def add_posting(self,term:str,doc_id:int)->None:
        if term not in self.index:
            self.index[term] = {}
        if doc_id in self.index[term]:
            self.index[term][doc_id] += 1
        else:
            self.index[term][doc_id] = 1
            
    def __exit__(self, *exc_info):
        if self.flag == "n":
            print("updating index...")
            self.update(self.index)
            #print("updating collection...")
            #self.update(self.collection)
            print("commiting...")
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)
            

if __name__ == "__main__":
    with open('collection.tsv', 'r',encoding='utf-8') as file:
        with InvertedIndex('inverted_index.sqlite', new=True,) as index:
            lengths = {}
            for i,line in tqdm(enumerate(file)):
                fields = line.strip().split('\t')
                terms = preprocess(fields[1])
                doc_id = int(fields[0])
                lengths[doc_id] = len(terms)
                #index.collection[doc_id] = terms
                for term in terms:
                    if term is None:continue
                    index.add_posting(term, doc_id)