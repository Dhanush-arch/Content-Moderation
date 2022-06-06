import time
from collections import defaultdict
from tqdm import tqdm
import nltk
import random
import string
import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import torchtext
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentiment(model, sentence, vocab, unk_idx=0):
    """
    model is your PyTorch model
    sentence is string you wish to predict sentiment on
    vocab is dictionary, keys = tokens, values = index of token
    unk_idx is the index of the <unk> token in the vocab
    """
    tokens = sentence.split() #convert string to tokens, needs to be same tokenization as training
    indexes = [vocab.get(t, unk_idx) for t in tokens] #converts to index or unk if not in vocab
    tensor = torch.LongTensor(indexes).unsqueeze(0) #convert to tensor and add batch dimension
    output = model(tensor, batch_size=1) #get output from model
    # prediction = torch.sigmoid(output) #squeeze between 0-1 range
    output = output.squeeze().tolist()
    return output

class LSTM_AttentionModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, pre_train, embedding_tune):
        super(LSTM_AttentionModel, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weights = nn.Parameter(weights, requires_grad=embedding_tune)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits

def loadModel():
    model = torch.load('./Models/FullbinaryModel.pth').to(device)
    # model.eval()
    return model

def read_raw_text(line, input_keywords, euphemism_answer):
    start = time.time()
    all_text = []
    temp = line.split()
    if any(ele in temp for ele in euphemism_answer.keys()) and len(line) <= 150:
        all_text.append(line.strip())
    if len(all_text) == 0:
        if any(ele in temp for ele in input_keywords) and len(line) <= 150:
            all_text.append(line.strip())
    return all_text

def read_input_and_ground_truth(target_category_name):
    fname_euphemism_answer = './data/euphemism_answer_' + target_category_name + '.txt'
    fname_target_keywords_name = './data/target_keywords_' + target_category_name + '.txt'
    euphemism_answer = defaultdict(list)
    with open(fname_euphemism_answer, 'r') as fin:
        for line in fin:
            ans = line.split(':')[0].strip().lower()
            for i in line.split(':')[1].split(';'):
                euphemism_answer[i.strip().lower()].append(ans)
    input_keywords = sorted(list(set([y for x in euphemism_answer.values() for y in x])))
    target_name = {}
    count = 0
    with open(fname_target_keywords_name, 'r') as fin:
        for line in fin:
            for i in line.strip().split('\t'):
                target_name[i.strip()] = count
            count += 1
    return euphemism_answer, input_keywords, target_name


def has_drugs_name(input_text, input_keywords, euphemism_answer):
    all_text = read_raw_text(input_text, input_keywords, euphemism_answer)
    has_drugs = True if len(all_text) > 0 else False
    return has_drugs, all_text
