import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import pandas as pd
import re, os

# Set environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def preprocess(doc: str) -> str:
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
    ])
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")

# Function to rewrite queries using T5
def rewrite_query_with_context(queries_df):
    rewritten_queries = []
    for i, row in queries_df.iterrows():
        # Get all previous turns of the conversation
        conversation_history = queries_df[(queries_df['topic_number'] == row['topic_number']) & 
                                          (queries_df['turn_number'] < row['turn_number'])]
        # Concatenate conversation history and current query
        context_input = "|||".join(conversation_history['query'].tolist() + [row['query']])
        # Encode the input context
        input_ids = tokenizer.encode(context_input, return_tensors="pt")
        # Generate the rewritten query
        outputs = model.generate(input_ids, max_length=512)
        # Decode the generated id to text
        rewritten_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Store the rewritten query
        rewritten_queries.append(preprocess(rewritten_query))
    return rewritten_queries


# Initialize PyTerrier
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

# Load the index 
index_loc = "./terrier_index"
index = pt.IndexFactory.of(index_loc)

# Load the rerankers
monoT5 = MonoT5ReRanker(verbose=True, batch_size=32)
duoT5 = DuoT5ReRanker(verbose=True, batch_size=32)

# Define the pipelines
bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k1": 1.5, "bm25.b": 0.5})
mono_pipeline =  bm25 >> pt.text.get_text(index, "text") >> monoT5
duo_pipeline = mono_pipeline % 50 >> duoT5 

# Load queries and qrels
queries_file = "./data/queries_test.csv"
qrels_file = "./data/qrels_train.txt"
queries = pd.read_csv(queries_file, encoding="utf-8")
qrels = pt.io.read_qrels(qrels_file)

queries['query2'] = rewrite_query_with_context(queries)
print(queries['query2'])
queries['query'] = queries['query2']

# Run the pipeline
mono_results = mono_pipeline.transform(queries[['qid', 'query']])
duo_results = duo_pipeline.transform(queries[['qid', 'query']])



top_50_duo_results = duo_results.head(50)

mono_results = mono_results[~mono_results['docno'].isin(top_50_duo_results['docno'])]

combined_results = pd.concat([top_50_duo_results, mono_results])

pt.io.write_results(combined_results, "./data/test.txt")

