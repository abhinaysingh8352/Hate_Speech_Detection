import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import LSTM, Embedding, SpatialDropout1D, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# Loading datasets
df_twitter = pd.read_csv('train.csv')
df_offensive = pd.read_csv('labeled_data.csv')

# Drop unnecessary columns
df_twitter.drop('id', axis=1, inplace=True)
df_offensive.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither', 'class'], axis=1, inplace=True)

# Rename class values for better readability
df_offensive['class'].replace((0, 1, 2), ('hate_speech', 'offensive_language', 'neither'), inplace=True)
df_offensive.rename(columns={'class': 'label'}, inplace=True)

# Combine the datasets
frames = [df_twitter, df_offensive]
df = pd.concat(frames)

# Initialize stemmer and stopwords
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Corrected invalid escape sequence
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Corrected invalid escape sequence
    text = re.sub(r'<.*?>+', '', text)  # Corrected invalid escape sequence
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    text = [word for word in text.split(' ') if word not in stopword]  # Remove stopwords
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]  # Stemming
    text = " ".join(text)
    return text

# Apply cleaning to tweets
df['tweet'] = df['tweet'].apply(clean_text)

# Data visualization
sns.countplot('label', data=df)
print(df.shape)

# Split data into train and test sets
x = df['tweet']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Text vectorization
count = CountVectorizer(stop_words='english', ngram_range=(1, 5))
x_train_vectorizer = count.fit_transform(x_train)
x_test_vectorizer = count.transform(x_test)

# Parameters for LSTM model
max_words = 50000
max_len = 300

# Tokenization
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

# Build LSTM model
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
checkpoint = ModelCheckpoint(filepath='./model_checkpoint.h5', save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

# Preprocessing test data for evaluation
test_sequences = tokenizer.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

# Train the model
accr = model.evaluate(test_sequences_matrix, y_test)
lstm_prediction = model.predict(test_sequences_matrix)

# Making predictions
res = []
for prediction in lstm_prediction:
    res.append(0 if prediction[0] < 0.5 else 1)

# Print confusion matrix
print(confusion_matrix(y_test, res))

# Save the tokenizer using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the model
from keras.models import load_model
model = load_model('model.h5')

# Function to preprocess text before prediction
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = sequence.pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

# Predict sentiment function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    return 'Positive' if prediction[0][0] > 0.5 else 'Negative'

# Example usage
text = "I love this product!"
print(predict_sentiment(text))

# Load the trained model and tokenizer for further use
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
loaded_model = load_model('hate&abusive_model.h5')

# Test a new sentence
test_text = 'I hate my country'
seq = tokenizer.texts_to_sequences([test_text])
padded_sequence = sequence.pad_sequences(seq, maxlen=max_len)

# Make a prediction with the loaded model
pred = loaded_model.predict(padded_sequence)

# Output the prediction result
if pred > 0.5:
    print("Hate")
else:
    print("No hate")
