#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import urllib
# Get the list of stopwords
import string
from ekphrasis.classes.preprocessor import TextPreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
# ! pip install emoji
import emoji
import math
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
# Initialize WordNet Lemmatizer
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")
import sys
from nltk.tokenize import word_tokenize

tt = nltk.tokenize.TweetTokenizer()

punct_set = set(string.punctuation + '''…'"`’”“'''  + '️')
stopwords_list = nltk.corpus.stopwords.words('english')
punct_set = set(string.punctuation)
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
stemmer = PorterStemmer()
def my_preprocessor(text, tokenize=word_tokenize,hashtag_segmenter=hashtag_segmenter, punct_set=punct_set, stoplist=stopwords_list):
    text = text.lower()
    text = emoji.demojize(text)
    # Tokenize
    text = text.split()
    text = ' '.join(text)
    tokens = tt.tokenize(text)
    # print(tokens)
    updated_tokens = []
    for t in tokens:
        # if t in emoji.EMOJI_DATA:
        #     updated_tokens += emoji.demojize(t).strip(':').split("_")
        if t.startswith('@'):
            pass
        elif t.startswith('#'):
            # updated_tokens += hashtag_segmenter.pre_process_doc(t).split()
            updated_tokens.append(t)
        # Remove URLs because we will get them from the expanded_urls field anyways
        elif t.startswith('http'):
            parsed_url = urllib.parse.urlparse(t)
            updated_tokens.append(parsed_url.netloc)
        else:
            lemma = lemmatizer.lemmatize(t)
            # skip stopwords and empty strings, include anything else
            if lemma and lemma not in stoplist and lemma not in punct_set:
                updated_tokens.append(lemma)
    return ' '.join(updated_tokens)

data = pd.read_csv("CL-II-MisinformationData - Sheet1.csv")
# data.head()


X_train, X_test_temp, y_train, y_test_temp = train_test_split(data[['tweet']], data['label'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_temp, y_test_temp, test_size=0.5, random_state=0)



# train = pd.concat([X_train, y_train], axis=1)
# val = pd.concat([X_val, y_val], axis=1)
# test = pd.concat([X_test, y_test], axis=1)
train = X_train.merge(y_train, left_index = True, right_index = True)
val = X_val.merge(y_val, left_index = True, right_index = True)
test = X_test.merge(y_test, left_index = True, right_index = True)

train.to_csv("train.csv")
val.to_csv("val.csv")
test.to_csv("test.csv")






label_encoder = LabelEncoder()
# Fit label encoder and transform y
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_val = label_encoder.fit_transform(y_val)

X_train['tweet'] = X_train['tweet'].apply(my_preprocessor)
X_test['tweet'] = X_test['tweet'].apply(my_preprocessor)
X_val['tweet'] = X_val['tweet'].apply(my_preprocessor)


vectorizer = TfidfVectorizer()
# get tf-df values
tfidf_matrix_train = vectorizer.fit_transform(X_train['tweet'])
tfidf_matrix_test = vectorizer.transform(X_test['tweet'])
tfidf_matrix_val = vectorizer.transform(X_val['tweet'])
# tfidf_matrix_train.shape


with open('tfidf_vectorizer_Xtrain.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix_train, file)
with open('tfidf_vectorizer_Xtest.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix_test, file)
with open('tfidf_vectorizer_Xval.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix_val, file)
with open('tfidf_vectorizer_Ytrain.pkl', 'wb') as file:
    pickle.dump(y_train, file)
with open('tfidf_vectorizer_Ytest.pkl', 'wb') as file:
    pickle.dump(y_test, file)
with open('tfidf_vectorizer_Yval.pkl', 'wb') as file:
    pickle.dump(y_val, file)


train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")




training_data = []
for index, row in train.iterrows():
    label = row['label']
    tweet = row['tweet']
    tweet = my_preprocessor(tweet)
    training_data.append(f'__label__{label} {tweet.strip()}')
        # training_data.append(f'{tweet.strip()}')

# with open('train.txt', 'w') as f:
#     f.write('\n'.join(training_data))
validation_data = []
for index, row in val.iterrows():
    label = row['label']
    tweet = row['tweet']
    tweet = my_preprocessor(tweet)
    # if (len(tweet.split()) <= 150):
    validation_data.append(f'__label__{label} {tweet.strip()}')
testing_data = []
for index, row in test.iterrows():
    
    label = row['label']
    tweet = row['tweet']
    tweet = my_preprocessor(tweet)
    # if (len(tweet.split()) <= 150):
    testing_data.append(f'__label__{label} {tweet.strip()}')

with open('train.txt', 'w') as f:
    f.write('\n'.join(training_data))
 
with open('val.txt', 'w') as f:
    f.write('\n'.join(validation_data))
with open('test.txt', 'w') as f:
    f.write('\n'.join(testing_data))
with open('train_val.txt', 'w') as f:
    f.write('\n'.join(training_data))
    f.write('\n'.join(validation_data))




import fasttext
model_train = fasttext.train_unsupervised('train_val.txt', dim = 100)
model_train.save_model("fasttext_model.bin")
# model.get_word_vector()



from scipy.sparse import csr_matrix
def parse_sentence(sentence):
    parts = sentence.split(' ', 1)
    label = parts[0].replace('__label__', '')
    text = parts[1]
    return label, text

def sentence_to_matrix(sentence, model, max_len):
    # Tokenize the sentence into words
    words = sentence.split()
    # Initialize an empty matrix to store word embeddings
    matrix = np.zeros((max_len, model.get_dimension()))  # Initialize with zeros
    # Iterate over each word and get its embedding
    for i, word in enumerate(words[:max_len]):
        matrix[i] = model.get_word_vector(word)
    return matrix
parsed_sentences = []
with open('train.txt') as f:
    for line in f.readlines():
        parsed_sentences.append(line)
parsed_sentences = [parse_sentence(sentence) for sentence in parsed_sentences]
labels = [label for label, _ in parsed_sentences]
texts = [text for _, text in parsed_sentences]

# Determine the maximum length of sentences in the dataset
max_len = max(len(sentence.split()) for sentence in texts)
max_len =140
sentence_matrices = []
# max_len = 100
for text in texts:
    matrix = sentence_to_matrix(text, model_train, max_len)
    sentence_matrices.append(matrix)
    
# # Convert labels to binary format
# label_dict = {'real': 1, 'fake': 0}
# labels_binary = [label_dict[label] for label in labels]

# Convert the list of matrices to a numpy array
sentence_matrices_train = np.array(sentence_matrices)

# # Convert labels to numpy array
# labels_binary = np.array(labels_binary)

# print("Shape of sentence matrices:", sentence_matrices_train.shape)
# print("Shape of labels:", labels_binary.shape)



with open('sentence_matrices_train.pkl', 'wb') as file:
    pickle.dump(sentence_matrices_train, file)
# len(sentence_matrices)



parsed_sentences = []
with open('val.txt') as f:
    for line in f.readlines():
        parsed_sentences.append(line)


parsed_sentences = [parse_sentence(sentence) for sentence in parsed_sentences]
labels = [label for label, _ in parsed_sentences]
texts = [text for _, text in parsed_sentences]

sentence_matrices = []
# max_len = 100
for text in texts:
    matrix = sentence_to_matrix(text, model_train, max_len)
    sentence_matrices.append(matrix)
    
# # Convert labels to binary format
# label_dict = {'real': 1, 'fake': 0}
# labels_binary = [label_dict[label] for label in labels]

# Convert the list of matrices to a numpy array
sentence_matrices_val = np.array(sentence_matrices)

# # Convert labels to numpy array
# labels_binary = np.array(labels_binary)

# print("Shape of sentence matrices:", sentence_matrices_val.shape)
# print("Shape of labels:", labels_binary.shape)


with open('sentence_matrices_val.pkl', 'wb') as file:
    pickle.dump(sentence_matrices_val, file)


parsed_sentences = []
with open('test.txt') as f:
    for line in f.readlines():
        parsed_sentences.append(line)

parsed_sentences = [parse_sentence(sentence) for sentence in parsed_sentences]
labels = [label for label, _ in parsed_sentences]
texts = [text for _, text in parsed_sentences]

sentence_matrices = []
# max_len = 100
for text in texts:
    matrix = sentence_to_matrix(text, model_train, max_len)
    sentence_matrices.append(matrix)
    
# # Convert labels to binary format
# label_dict = {'real': 1, 'fake': 0}
# labels_binary = [label_dict[label] for label in labels]

# Convert the list of matrices to a numpy array
sentence_matrices_test = np.array(sentence_matrices)

# # Convert labels to numpy array
# labels_binary = np.array(labels_binary)
# print("Shape of sentence matrices:", sentence_matrices_test.shape)
# print("Shape of labels:", labels_binary.shape)




with open('sentence_matrices_test.pkl', 'wb') as file:
    pickle.dump(sentence_matrices_test, file)


def preprocess(testfile):
    data = pd.read_csv(testfile)
    # X_test = data['tweet']
    # y_test = data['label']
    y_new_test = label_encoder.fit_transform(data['label'])
    data['New_tweet'] = data['tweet'].apply(my_preprocessor)
    tfidf_matrix_new_test = vectorizer.transform(data['New_tweet'])
    with open('tfidf_matrix_new_test.pkl', 'wb') as file:
        pickle.dump(tfidf_matrix_new_test, file)
    with open('y_new_test.pkl', 'wb') as file:
        pickle.dump(y_new_test, file)
        for index, row in data.iterrows():
            label = row['label']
            tweet = row['tweet']
            tweet = my_preprocessor(tweet)
            # if (len(tweet.split()) <= 150):
            testing_data.append(f'__label__{label} {tweet.strip()}')

    with open('test_new.txt', 'w') as f:
        f.write('\n'.join(testing_data))
 
    loaded_model = fasttext.load_model("fasttext_model.bin")

    parsed_sentences = []
    with open('test_new.txt') as f:
        for line in f.readlines():
            parsed_sentences.append(line)

    parsed_sentences = [parse_sentence(sentence) for sentence in parsed_sentences]
    labels = [label for label, _ in parsed_sentences]
    texts = [text for _, text in parsed_sentences]

    sentence_matrices = []
    # max_len = 100
    for text in texts:
        matrix = sentence_to_matrix(text, loaded_model, max_len)
        sentence_matrices.append(matrix)
        
    # # Convert labels to binary format
    # label_dict = {'real': 1, 'fake': 0}
    # labels_binary = [label_dict[label] for label in labels]

    # Convert the list of matrices to a numpy array
    sentence_matrices_new_test = np.array(sentence_matrices)

    with open('sentence_matrices_new_test.pkl', 'wb') as file:
        pickle.dump(sentence_matrices_new_test, file)
