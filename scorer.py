import math
from typing import Dict, List, Tuple
from collections import UserDict as DictClass
from collections import defaultdict
from collections import Counter
import abc

CollectionType = Dict[str, List[int]]

class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_length(self) -> int:
        """Total number of terms for all documents."""
        return sum(len(docs) for docs in self.values())

    def avg_length(self) -> float:
        """Average number of terms across all documents."""
        return self.total_length() / len(self)

class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
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

        for term, query_freq in query_term_freqs.items():
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
        collection: DocumentCollection,
        index: CollectionType,
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        if not term in self.index:
            return
        N = len(self.collection)
        nt = len(self.index.get(term) or [])          
        avgdl = self.collection.avg_length()
        for doc_id in self.collection:
            doc = self.collection.get(doc_id)
            dlen = len(doc)
            ctd = doc.count(term)
            self.scores[doc_id] += ((ctd*(1+self.k1)) / (ctd+self.k1*(1-self.b+self.b*(dlen/avgdl)))) * math.log(N/nt)