# %%
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
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch



# %%
filename = sys.argv[1]
data = pd.read_csv(filename)
# data.head()


# %%


X_train, X_test_temp, y_train, y_test_temp = train_test_split(data[['tweet']], data['label'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_temp, y_test_temp, test_size=0.5, random_state=0)


# %%
punct_set = list(set(string.punctuation))
punct_set += ['’','•']

stopwords_list = nltk.corpus.stopwords.words('english')
punct_set = set(string.punctuation)
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
stemmer = PorterStemmer()
# print(punct_set)
# print(stopwords_list)

# %%
from nltk.tokenize import word_tokenize
tt = nltk.tokenize.TweetTokenizer()
# print(punct_set)
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


# %%
# print(my_preprocessor("• 3 On the 15/03 NCDC directly contacted a Twitter user who mentioned his friend who returned from UK had runny nose but could not reach authorities for testing. Within 12 hours of communication with us via DM a sample was collected. We’re committed to doing our best https://t.co/fccdGij3uG,real"))

# %%

X_train['tweet'] = X_train['tweet'].apply(my_preprocessor)
X_test['tweet'] = X_test['tweet'].apply(my_preprocessor)
X_val['tweet'] = X_val['tweet'].apply(my_preprocessor)


# %%
# temp = X_train['tweet'].tolist()
# # print(temp[0:10])
# for i in range(10):
#     print(temp[i])
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
with open('tfidf_vectorizer_Ytrain.pkl', 'wb') as file:
    pickle.dump(y_train, file)
with open('tfidf_vectorizer_Ytest.pkl', 'wb') as file:
    pickle.dump(y_test, file)
with open('tfidf_vectorizer_Yval.pkl', 'wb') as file:
    pickle.dump(y_val, file)

