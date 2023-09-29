import math
from typing import Dict, List, Tuple
from collections import UserDict as DictClass
from collections import defaultdict
from collections import Counter
import abc
from tqdm import tqdm
from indexing import preprocess,InvertedIndex
CollectionType = Dict[str, List[int]]

class DocumentCollection(DictClass):
    def total_length(self) -> int:
            """Total number of terms for all documents."""
            return sum(self.values())

    def avg_length(self) -> float:
        """Average number of terms across all documents."""
        return self.total_length() / len(self)

class Scorer(abc.ABC):
    def __init__(
        self,
        index: CollectionType,
        collection: DocumentCollection
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information. List of doc ids
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def get_top_n_results(self, n: int) -> List[Tuple[int, float]]:
        return list(sorted(self.scores.items(), key=lambda item: item[1], reverse=True))[:1000]

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in tqdm(query_term_freqs.items()):
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError
    
class ScorerBM25(Scorer):
    def __init__(
        self,
        index: CollectionType,
        collection: DocumentCollection,
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(index, collection)
        print(collection.values())
        print(index.values())
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        docs = self.index.get(term) or []
        N = self.collection.total_length()
        nt = len(docs)
        avgdl = self.collection.avg_length()
        for doc_id, ctd in docs.items():
            dlen = self.collection.get(doc_id) or 0
            self.scores[doc_id] += ((ctd*(1+self.k1)) / (ctd+self.k1*(1-self.b+self.b*(dlen/avgdl)))) * math.log(N/nt) 


if __name__ == "__main__":
    print("Getting collection...")
    lengths = {}
    with open('collection.tsv', 'r',encoding='utf-8') as file:
        for i,line in tqdm(enumerate(file)):
            if i == 10000: break
            fields = line.strip().split('\t')
            terms = preprocess(fields[1])
            doc_id = int(fields[0])
            lengths[doc_id] = len(terms)
    print("Scoring...")
    index_file = "inverted_index.sqlite"
    index = InvertedIndex(index_file)
    scorer_bm25 = ScorerBM25(index.index, DocumentCollection(lengths))
    print("Reading queries...")
    with open('queries_train.csv','r') as f:
        next(f)
        index = 1
        print("Writing queries...")
        with open("output.txt","w") as trec:
            for line in tqdm(f):
                if index==2:break
                columns = line.split(',')
                turn_indentifier = columns[0]
                placeholder = "Q0"
                test = scorer_bm25.score_collection(preprocess(columns[1]))
                top_results = scorer_bm25.get_top_n_results(1000)
                print("top_results")
                print(top_results)
                for i, res in enumerate(top_results):
                    doc_id = res[0]
                    doc_score = res[1]
                    line = f"{turn_indentifier} {placeholder} {doc_id} {i+1} {doc_score} BM25\n"
                    print("result written")
                    trec.write(line)
                index+=1