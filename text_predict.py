import pandas as pd
from ipywidgets import Text, Label
from IPython.display import display, clear_output
import ipywidgets as widgets
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam


sentences_file = "./sample_sentences.csv"
df = pd.read_csv(sentences_file)
sentences = df[0].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1 

input_sequences = []
for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

X , y = input_sequences[:, :-1], input_sequences[:, -1]

model = Sequential([
    Embedding(total_words, 50, input_length=max_seq_length-1), 
    LSTM(128, return_sequences=True), 
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0) 

# Function to predict next word
def predict_next_words(input_text, top_k=3):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
    predictions = model.predict(token_list, verbose=0)[0]

    predicted_indices = np.argsort(predictions)[-top_k:][::-1]
    predicted_words = [word for word, index in tokenizer.word_index.items() if index in predicted_indices]

    return ', '.join(predicted_words) if predicted_words else "No suggestions"
    
inp = Text(placeholder='Type...')
out = Label()

def update(change):
    clear_output(wait=True)
    display(inp, out)
    out.value = predict_next_words(change['new']) if change['new'].strip() else ""

inp.observe(update, names='value')
display(inp, out)

