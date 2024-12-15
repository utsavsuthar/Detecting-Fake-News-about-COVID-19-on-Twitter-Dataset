
from sklearn.model_selection import train_test_split
from math import log10
import emoji
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
import pickle
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
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
import os
lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")
import sys
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
filename = sys.argv[1] 
df = pd.read_csv(filename)
punct_set = list(set(string.punctuation))
punct_set += ['’','•']

stopwords_list = nltk.corpus.stopwords.words('english')
punct_set = set(string.punctuation)
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
stemmer = PorterStemmer()
# print(punct_set)
# print(stopwords_list)
from nltk.tokenize import word_tokenize
tt = nltk.tokenize.TweetTokenizer()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def my_preprocessor(text, tokenize=word_tokenize,hashtag_segmenter=hashtag_segmenter, punct_set=punct_set, stoplist=stopwords_list):
    text = text.lower()
    # text = emoji.demojize(text)
    # Tokenize
    text = text.split()
    text = ' '.join(text)
    tokens = tt.tokenize(text)
    tokens = [word.replace("’", "") and word.replace("•", "") for word in tokens]
    # print(tokens)
    # print(tokens)
    updated_tokens = []
    for t in tokens:
        if t in emoji.EMOJI_DATA:
            updated_tokens += emoji.demojize(t).strip(':').split("_")
        if t.startswith('@') or t.isdigit()  or t in punct_set or t in stoplist:
            # print(t)
            pass
        elif t.startswith('#'):
            updated_tokens += hashtag_segmenter.pre_process_doc(t).split()
            # updated_tokens.append(t)
        # Remove URLs because we will get them from the expanded_urls field anyways
        elif t.startswith('http'):
            # parsed_url = urllib.parse.urlparse(t)
            # updated_tokens.append(parsed_url.netloc)
            # updated_tokens.append('link')
            pass
        else:
            lemma = lemmatizer.lemmatize(t)
            # skip stopwords and empty strings, include anything else
            if lemma:
                updated_tokens.append(lemma)
    return ' '.join(updated_tokens)

df['preprocessed_tweet'] = df['tweet'].apply(my_preprocessor)
# df['preprocessed_tweet'] = df['tweet'].apply(my_preprocessor)
# print(df[5:10])


# In[8]:


# print(df["tweet"][3])
# print(df["preprocessed_tweet"][3])
# print(final_output("#IndiaFightsCorona"))


# In[9]:


X_test = df['preprocessed_tweet']
y_test = df['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=49)


# In[10]:


# Create DataFrames for train, test, and validation sets
# train_df = pd.DataFrame({'X': X_train, 'y': y_train})
test_df = pd.DataFrame({'tweet': X_test, 'label': y_test})
# val_df = pd.DataFrame({'X': X_val, 'y': y_val})
# Export DataFrames to CSV files
# train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test.csv', index=False)
# val_df.to_csv('validate_data.csv', index=False)


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
import pickle
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel


# In[2]:


def select_bert_model():
    print("Choose a BERT model:")
    print("1. bert-base-cased")
    print("2. bert-base-uncased")
    print("3. covid-twitter-bert")
    print("4. twhin-bert-base")
    print("5. socbert")
    choice = input("Enter the number corresponding to your choice (1-5): ")
    return choice


# In[4]:


choice = select_bert_model()
if choice == '1':
    bert_model = 'bert-base-cased'
    bert_model2 = 'bert-base-cased'
elif choice == '2':
    bert_model = 'bert-base-uncased'
    bert_model2 = 'bert-base-uncased'
elif choice == '3':
    bert_model = 'digitalepidemiologylab/covid-twitter-bert'
    bert_model2 = 'covid-twitter-bert'
elif choice == '4':
    bert_model = 'Twitter/twhin-bert-base'
    bert_model2 = 'twhin-bert-base'
elif choice == '5':
    bert_model = 'sarkerlab/SocBERT-base'
    bert_model2 = 'socbert'
else:
    print("Invalid choice. Please enter a number between 1 and 5.")
    exit()



def bert_base_model(sentences,bert_model):
    # Batch size
    batch_size = 32
    sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    all_embeddings = []

    # Load pre-trained BERT model and tokenizer

    if bert_model == 'digitalepidemiologylab/covid-twitter-bert' or 'Twitter/twhin-bert-base' or 'sarkerlab/SocBERT-base':
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModel.from_pretrained(bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        model = BertModel.from_pretrained(bert_model)
   
    for batch in sentence_batches:
        # Tokenize and encode the batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Forward pass to get the representations
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the output embeddings (CLS token)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Append the embeddings to the list
        all_embeddings.append(embeddings)

    # Concatenate the embeddings from all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)

    # Print the shape of the final embeddings
    print("Shape of final embeddings:", final_embeddings.shape)
    return final_embeddings


df_test = pd.read_csv('test.csv')
print("Creating Test Embeddings, Please wait...")
X_test_embeddings = bert_base_model(df_test['tweet'].tolist(),bert_model)
# print(X_test_embeddings.shape)



# torch.save(X_val_embeddings, f'X_val_{bert_model2}.pt')
torch.save(X_test_embeddings, f'X_test_{bert_model2}.pt')
# torch.save(X_train_embeddings, f'X_train_{bert_model2}.pt')
label_encoder = LabelEncoder()
# Fit label encoder and transform y

y_test = label_encoder.fit_transform(df_test['label'])

with open('tfidf_vectorizer_Ytest.pkl', 'wb') as file:
    pickle.dump(y_test, file)

current_directory = os.path.dirname(os.path.abspath(__file__))
import subprocess
# models = ['DNN', 'CNN', 'LSTM']
subprocess.Popen(["python3", os.path.join(current_directory, 'DNN.py'), bert_model2])
    # eval_process.wait()
subprocess.Popen(["python3", os.path.join(current_directory, 'CNN.py'),bert_model2])
subprocess.Popen(["python3", os.path.join(current_directory, 'Automodel.py'),bert_model2])
print("All subprocesses completed.")





