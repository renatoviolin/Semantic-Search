# %%
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
model = hub.load(module_url)
batch_size = 16


def embed(input):
    return model(input)


def get_scores(input_query, input_corpus, topk=5):
    n_samples = len(input_corpus)
    emb = np.zeros([n_samples, 512])
    num_batches = n_samples // batch_size
    for i in range(num_batches + 1):
        start = batch_size * i
        end = (batch_size * i) + batch_size
        emb[start:end] = embed(input_corpus[start:end])

    emb_query = embed([input_query])[0]
    input_matrix = np.vstack([[emb_query] * n_samples])

    results = np.dot(input_matrix, emb.T)[0]
    topk = results.argsort()[-topk:][::-1]
    scores = results[topk]
    sentences = [input_corpus[idx] for idx in topk]
    return [str(s) for s in scores], sentences


# %%
# lines = []
# with open('dataset.txt') as f:
#     lines = f.readlines()
# query_text = 'time sharing operational system'

# s, sen = embed([query_text])