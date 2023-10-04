# dat640_group_project
First it is important to download the dataset from: https://gustav1.ux.uis.no/dat640/msmarco-passage.tar.gz
The dataset is large, hence it is not in the repository

# Indexing
The indexing.py must be run first. It will create a terrier index for the dataset.

# Retreival
The scoring.py must be run after the index is created. This will utilize the terrier index and use BM25 to score the documents

# Scores
The command: ./eval/trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 qrels_train.txt bm25score.txt
Will give out the Recall@1000, NDCG@3, MAP, and MRR.
