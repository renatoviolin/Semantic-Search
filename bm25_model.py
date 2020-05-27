from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def word_token(tokens, lemma=False):
    tokens = str(tokens)
    tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
    tokens = re.sub(r"\s+", " ", tokens)
    if lemma:
        return " ".join([lemmatizer.lemmatize(token, 'v') for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])
    else:
        return " ".join([token for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])


def get_scores(input_query, input_corpus, topk=5):
    docs = [input_query] + input_corpus
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus[1:])

    input_query = tokenized_corpus[0]
    bm25_scores = bm25.get_scores(input_query)

    bm25_scores = [(i, v) for i, v in enumerate(bm25_scores)]

    bm25_scores.sort(key=lambda x: x[1], reverse=True)

    idx = np.array(bm25_scores)[:topk, 0].astype(int)
    scores = np.array(bm25_scores)[:topk, 1]
    sentences = [input_corpus[i] for i in idx]
    return scores, sentences


# #%%
# with open('dataset-pt.txt') as f:
#     lines = f.readlines()


# %%
# scores, sent = get_scores('estrutura de repetição while', lines)
