# dat640_group_project
## Prerequisites
First it is important to download the dataset from: https://gustav1.ux.uis.no/dat640/msmarco-passage.tar.gz
The dataset is large, hence it is not in the repository <br/>
Also need to to install the required pip packages
```shell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

## Indexing
There is two different indexing files, one for the baseline and one with document expansion.<br>
One of the indexing files must be run first. It will create a terrier index for the dataset.<br>

To run the baseline indexing
```shell
python indexingBaseline.py
```
To run the document expansion indexing
```shell
python indexingExpando.py
```

## Baseline Retreival
The baseline_retrival.py must be run after the baseline index is created. This will utilize the terrier index and use BM25 to retrieve the documents
```shell
python baseline_retrival.py
```

## Expando Mono Duo
The mono_duo_retrival.py must be run after the baseline index is created. This will utilize the terrier index and use Expando Mono Duo to retrieve the documents
```shell
python mono_duo_retrival.py
```


## Expando Mono Duo with T5 Query Rewrites
The final_retrival.py must be run after the document expansion index is created. This will utilize the terrier index and use Expando Mono Duo with query rewrites to retrieve the documents
```shell
python final_retrival.py
```

## Scores
To enable te trec_eval, you must cd into the folder trec_eval and then use the command make<br />
The command: 
```shell
./trec_eval/trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 data/qrels_train.txt {file}
```
Will give out the Recall@1000, NDCG@3, MAP, and MRR. Replace file with the file which are going to be scored.
- {./data/bm25score.txt} for the baseline results
- {./data/monoduo.txt} for the Expando Mono Duo results
- {./data/finalretrival.txt} for the Expando Mono Duo with T5 Query Rewrites results
