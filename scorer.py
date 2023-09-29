import math
from typing import Dict, List, Tuple
from collections import UserDict as DictClass
from collections import defaultdict
from collections import Counter
import abc
from tqdm import tqdm
from indexing import preprocess,InvertedIndex
CollectionType = Dict[str, List[int]]

class Scorer(abc.ABC):
    def __init__(
        self,
        index: CollectionType,
        lengths: Dict[int,int]
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
        self.lengths = lengths
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
        lengths: Dict[int,int],
        index: CollectionType,
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(lengths, index)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        docs = self.index.get(term) or []
        N = sum(self.lengths.values())
        nt = len(docs)
        avgdl = N / len(self.lengths)
        for doc_id, ctd in docs.items():
            dlen = self.lengths.get(doc_id)
            ctd = ctd
            self.scores[doc_id] += ((ctd*(1+self.k1)) / (ctd+self.k1*(1-self.b+self.b*(dlen/avgdl)))) * math.log(N/nt) 


if __name__ == "__main__":
    print("Scoring...")
    index_file = "inverted_index.sqlite"
    index = InvertedIndex(index_file)
    scorer_bm25 = ScorerBM25(index.lengths, index.index)
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