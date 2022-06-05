# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer


import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

df = pd.read_csv("./vectorizor/vectorizor.csv")

# MAX_FEATURES = 200000 
# vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
# vectorizer.adapt(df["comment_text"])
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df["comment_text"])

df = []

def init_toxicity_model():
    model = tf.keras.models.load_model('./Models/toxicity_updated.h5')    
    return model

def toxicity_predict(model, input_text):
    # input_text = vectorizer(input_text)
    input_text = tokenizer.texts_to_sequences([input_text])
    pad = sequence.pad_sequences(input_text, maxlen=100)
    res = model.predict(pad)
    # res = res[0]
    # maxVal = res[tf.argmax(res)]
    return res