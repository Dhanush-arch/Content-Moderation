from flask import Flask, jsonify, request, redirect, render_template
from Binary_model import *
from Toxicity_model import *

import flask
app = Flask(__name__)
toxicity_model = init_toxicity_model()
binary_classifier_model = loadModel()
binary_classifier_model.eval()

euphemism_answer, input_keywords, target_name = read_input_and_ground_truth("drug")

input_text = ""
return_msg = ""

def read_vocab():
    vocab = dict()
    with open("./data/vocabs", 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

vocabs = read_vocab()
# print(vocabs)

@app.route('/')
def index():
    return render_template('index.html', msg=return_msg, text=input_text)

@app.route('/api', methods=['POST'])
def predict():
    if request.form["review_post"].strip() == "":
        return redirect("/")
    global input_text
    global return_msg
    input_text = request.form["review_post"]
    has_drugs, text =  has_drugs_name(input_text, input_keywords, euphemism_answer)
    
    if has_drugs:
        print(text)
        out = predict_sentiment(binary_classifier_model,request.form["review_post"], vocabs, 0)
        res = toxicity_predict(toxicity_model, request.form["review_post"])
        print("Drug detection: ", out)
        if abs(out[0]) > abs(out[1]):
            return_msg = "By Classifier: The Message is displayed to the user. By Toxicity Model: "+ str(max(res[0]))
        else:
            return_msg = "By Classifier: Message is Flagged temporarily. By Toxicity Model: " + str(max(res[0]))
        # euphemism_identification(binary_classifier_model, euphemism_answer.keys(), text, euphemism_answer, input_keywords, target_name)
    else:
        res = toxicity_predict(toxicity_model, request.form["review_post"])
        print("toxic percent: ", max(res[0]))

        toxic_percent = max(res[0])

        if toxic_percent > 0.6:
            return_msg = "Message Blocked. The message is detected with more toxic comments"
        elif toxic_percent > 0.3:
            return_msg = "Message is Flagged temporarily. It is detected to have some toxic words"
        else:
            return_msg = "The Message is displayed to the user"

    return redirect("/")

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=80, debug=True)
    app.run()