from flask import Flask, jsonify, request, redirect, render_template
from Binary_model import *
from Toxicity_model import *

import flask
app = Flask(__name__)
toxicity_model = init_toxicity_model()
binary_classifier_model = loadModel()
binary_classifier_model.eval()

euphemism_answer, input_keywords, target_name = read_input_and_ground_truth("drug")

return_msg = ""

def read_vocab():
    vocab = dict()
    with open("./data/vocabs", 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

vocabs = read_vocab()
print(vocabs)

@app.route('/')
def index():
    return render_template('index.html', predict=return_msg)

@app.route('/api', methods=['POST'])
def predict():

    has_drugs, text =  has_drugs_name(request.form["review_post"], input_keywords, euphemism_answer)
    
    if has_drugs:
        print(text)
        print(predict_sentiment(binary_classifier_model,request.form["review_post"], vocabs, 0))
        # euphemism_identification(binary_classifier_model, euphemism_answer.keys(), text, euphemism_answer, input_keywords, target_name)
    else:
        res = toxicity_predict(toxicity_model, request.form["review_post"])

    global return_msg
    return_msg = ""
    return redirect("/")

if __name__ == '__main__':
    app.run(host='localhost', port=8081, debug=True)