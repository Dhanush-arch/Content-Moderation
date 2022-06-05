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


''' Main Function '''
def euphemism_identification(model, top_words, all_text, euphemism_answer, input_keywords, target_name):
    print('\n' + '*' * 40 + ' [Euphemism Identification] ' + '*' * 40)
    NGRAMS = 1
    final_test = get_final_test(euphemism_answer, top_words, input_keywords)
    _, test_data, _, train_data_pre, test_data_pre, _, _ = get_train_test_data(input_keywords, target_name, all_text, final_test, NGRAMS, train_perc=0.8)

    print('-' * 40 + ' [Coarse Binary Classifier] ' + '-' * 40)
    print('Model: ' + "LSTM_Attention func")

    train_iter, test_iter, Final_test_iter = train_initialization(train_data_pre, test_data_pre, test_data, target_name, IsPre=1)
    eval_model(model, train_iter,)
    # eval_model(model, test_iter, loss_fn)
    return

''' Neural Models '''
def train_initialization(train_data, test_data, Final_test, target_name, IsPre=0):
    def load_dataset(train_data, test_data, Final_test, embedding_length, batch_size):
        def get_dataset(a_data, fields):
            examples = []
            for data_i in tqdm(a_data):
                examples.append(torchtext.data.Example.fromlist([data_i[0], data_i[1]], fields))
            return examples

        tokenize = lambda x: x.split()
        TEXT = torchtext.data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=15)
        LABEL = torchtext.data.LabelField()
        fields = [("text", TEXT), ("label", LABEL)]
        train_data = get_dataset(train_data, fields)
        test_data = get_dataset(test_data, fields)
        Final_test = get_dataset(Final_test, fields)
        train_data = torchtext.data.Dataset(train_data, fields=fields)
        test_data = torchtext.data.Dataset(test_data, fields=fields)
        Final_test = torchtext.data.Dataset(Final_test, fields=fields)

        # TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=embedding_length))
        # LABEL.build_vocab(train_data)
        # word_embeddings = TEXT.vocab.vectors
        # vocab_size = len(TEXT.vocab)
        # print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        # print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        # print("Label Length: " + str(len(LABEL.vocab)))

        train_iter, test_iter, Final_test_iter = torchtext.data.Iterator.splits((train_data, test_data, Final_test), batch_size=batch_size, sort=False, repeat=False)
        return TEXT, 0, 0, train_iter, test_iter, Final_test_iter

    # output_size = 2 if IsPre else max(target_name.values())+1
    # learning_rate = 0.002
    # hidden_size = 256
    embedding_length = 100
    # epoch_num = 3 if IsPre else 10
    batch_size = 1
    # pre_train = True
    # embedding_tune = False

    _, _, _, train_iter, test_iter, Final_test_iter = load_dataset(train_data, test_data, [[x[0], 0] for x in Final_test], embedding_length, batch_size)
    # model = LSTM_AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, pre_train, embedding_tune)

    # model = model.to(device)
    # loss_fn = F.cross_entropy
    return train_iter, test_iter, Final_test_iter

def eval_model(model, val_iter):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            # if (text.size()[0] is not 32):
            #     continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text = text.to(device)
            target = target.to(device)
            prediction = model(text)
            print("prediction: ",prediction)

''' Utility Functions '''
def get_train_test_data(input_keywords, target_name, all_text, final_test, NGRAMS, train_perc):
    print('[utils.py] Constructing train and test data...')
    all_data = []
    all_data_pre = []
    final_test_data = []
    for i in all_text:
        temp = nltk.word_tokenize(i)
        for j, keyword in enumerate(input_keywords):  # Add positive labels that belong to input keywords.
            if keyword not in temp:
                continue
            all_data.append([temp, target_name[keyword]])
            all_data_pre.append([temp, 1])  # is one of the target keywords.
        temp_index = random.randint(0, len(temp)-1)
        if temp[temp_index] not in input_keywords:  # Add negative labels that NOT belong to input keywords.
            all_data_pre.append([temp, 0])  # is NOT one of the target keywords.
        for j, keyword in enumerate(final_test):  # Construct final_test_data
            if keyword not in temp:
                continue
            temp_index = temp.index(keyword)
            final_test_data.append([temp, j])  # final_test_data's label is the id number of final_test. For later final_out construction

    def _shuffle_and_balance(all_data, max_len):
        random.shuffle(all_data)
        data_len = defaultdict(int)
        all_data_balanced = []
        for i in all_data:
            if data_len[i[1]] == max_len:
                continue
            data_len[i[1]] += 1
            all_data_balanced.append(i)
        random.shuffle(all_data_balanced)
        train_data = all_data_balanced[:int(train_perc * len(all_data_balanced))]
        test_data = all_data_balanced[int(train_perc * len(all_data_balanced)):]
        return train_data, test_data

    train_data, test_data = _shuffle_and_balance(all_data, max_len=2000)
    train_data_pre, test_data_pre = _shuffle_and_balance(all_data_pre, max_len=min(100000, sum([x[1]==0 for x in all_data_pre]), sum([x[1]==1 for x in all_data_pre])))
    unique_vocab_dict, unique_vocab_list = build_vocab(train_data, NGRAMS, min_count=10)
    return train_data, test_data, final_test_data, train_data_pre, test_data_pre, unique_vocab_dict, unique_vocab_list


def get_final_test(euphemism_answer, top_words, input_keywords):
    final_test = {}
    for x in top_words:
        if x in euphemism_answer:
            if any(ele in euphemism_answer[x] for ele in input_keywords):
                final_test[x] = euphemism_answer[x]
            else:
                final_test[x] = ['None']
        else:
            final_test[x] = ['None']
    return final_test


def build_vocab(xlist, NGRAMS, min_count):
    vocabi2w = ['[SOS]', '[EOS]', '[PAD]', '[UNK]']  # A list of unique words
    seen = defaultdict(int)
    for i in range(len(xlist)):
        tokens = nltk.word_tokenize(xlist[i][0])
        tokens = tokens if NGRAMS == 1 else ngrams_iterator(tokens, NGRAMS)
        for token in tokens:
            seen[token] += 1
    vocabi2w += [x for x in seen if seen[x] >= min_count]
    vocabw2i = {vocabi2w[x]:x for x in range(len(vocabi2w))}
    return vocabw2i, vocabi2w
