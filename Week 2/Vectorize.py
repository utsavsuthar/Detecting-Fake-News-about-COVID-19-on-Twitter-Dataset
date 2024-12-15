# %%

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")
import sys
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch


def select_bert_model():
    print("Choose a BERT model:")
    print("1. bert-base-cased")
    print("2. bert-base-uncased")
    print("3. covid-twitter-bert")
    print("4. twhin-bert-base")
    print("5. socbert")
    choice = input("Enter the number corresponding to your choice (1-5): ")
    return choice


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


X_train = pd.read_csv("train.csv")
X_val = pd.read_csv("val.csv")
X_test = pd.read_csv("test.csv")



def bert_base_uncased_rep(sentences,bert_model):
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

print("Creating model Embeddings...Please wait!!!")
X_train_embeddings = bert_base_uncased_rep(X_train['tweet'].tolist(),bert_model)
print(X_train_embeddings.shape)
X_val_embeddings = bert_base_uncased_rep(X_val['tweet'].tolist(),bert_model)
print(X_val_embeddings.shape)
X_test_embeddings = bert_base_uncased_rep(X_test['tweet'].tolist(),bert_model)
print(X_test_embeddings.shape)


torch.save(X_val_embeddings, f'X_val_{bert_model2}.pt')
torch.save(X_test_embeddings, f'X_test_{bert_model2}.pt')
torch.save(X_train_embeddings, f'X_train_{bert_model2}.pt')
print("Model Embeddings has been created!!!")