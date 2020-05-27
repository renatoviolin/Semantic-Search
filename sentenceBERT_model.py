from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')


def embed(input):
    return model.encode(input)


def get_scores(input_query, input_corpus, topk=5):
    emb_corpus = np.array(model.encode(input_corpus))
    emb_query = np.array(model.encode([input_query]))
    results = cosine_similarity(emb_query, emb_corpus)[0]
    topk = results.argsort()[-topk:][::-1]
    scores = results[topk]
    sentences = [input_corpus[idx] for idx in topk]
    return [str(s) for s in scores], sentences


# %%
# lines = []
# with open('dataset.txt') as f:
#     lines = f.readlines()
# query_text = 'time sharing operational system'

# s, sen = get_scores(query_text, lines)
