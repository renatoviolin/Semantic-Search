import flask
from flask import Flask, request, render_template
import json
import use_model
import bm25_model
import sentenceBERT_model
import sentenceROBERTA_model
import infersent_model
import bert_model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_predictions', methods=['post'])
def get_prediction():
    corpus = []
    d = {}
    try:
        input_query = request.json['input_query']
        input_corpus = request.json['input_corpus']
        split_token = request.json['split_token']
        top_k = request.json['top_k']
        if split_token == '0':
            corpus = input_corpus.split('\n')
        elif split_token == '1':
            corpus = input_corpus.replace('\n', '').split('.')

        use_scores, use_sentences = use_model.get_scores(input_query, corpus, int(top_k))
        d['use_scores'] = list(use_scores)
        d['use_sentences'] = use_sentences

        bm25_scores, bm25_sentences = bm25_model.get_scores(input_query, corpus, int(top_k))
        d['bm25_scores'] = list(bm25_scores)
        d['bm25_sentences'] = bm25_sentences

        sentenceBERT_scores, sentenceBERT_sentences = sentenceBERT_model.get_scores(input_query, corpus, int(top_k))
        d['sentenceBERT_scores'] = list(sentenceBERT_scores)
        d['sentenceBERT_sentences'] = sentenceBERT_sentences

        infersent_scores, infersent_sentences = infersent_model.get_scores(input_query, corpus, int(top_k))
        d['infersent_scores'] = infersent_scores
        d['infersent_sentences'] = list(infersent_sentences)

        bert_scores, bert_sentences = bert_model.get_scores(input_query, corpus, int(top_k))
        d['bert_scores'] = bert_scores
        d['bert_sentences'] = list(bert_sentences)

        roberta_scores, roberta_sentences = sentenceROBERTA_model.get_scores(input_query, corpus, int(top_k))
        d['roberta_scores'] = roberta_scores
        d['roberta_sentences'] = list(roberta_sentences)

        return app.response_class(response=json.dumps(d), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
