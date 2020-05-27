from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def embed(input):
    emb = []
    for sentence in input:
        input_ids = torch.tensor(tokenizer.encode(sentence.lower(), add_special_tokens=True)[:512]).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)[0]
            res = torch.mean(outputs, dim=1).detach().cpu().numpy()
        emb.append(res[0])
    return np.array(emb)


def get_scores(input_query, input_corpus, topk=5):
    emb_corpus = embed(input_corpus)
    emb_query = embed(input_query)
    results = cosine_similarity(emb_query, emb_corpus)[0]
    topk = results.argsort()[-topk:][::-1]
    scores = results[topk]
    sentences = [input_corpus[idx] for idx in topk]
    return [str(s) for s in scores], sentences


# %%
# lines = []
# with open('dataset.txt') as f:
#     lines = f.readlines()
# query_text = 'apple revenue'

# s, sen = get_scores(query_text, lines)


