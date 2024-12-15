#!/usr/bin/env python
# coding: utf-8

# In[59]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch.nn.functional as F



# In[60]:


import pickle
# Load TF-IDF vectorizer from file
with open('sentence_matrices_train.pkl', 'rb') as file:
    sentence_matrices_train = pickle.load(file)
with open('tfidf_vectorizer_Ytrain.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open('sentence_matrices_val.pkl', 'rb') as file:
    sentence_matrices_val = pickle.load(file)
with open('tfidf_vectorizer_Yval.pkl', 'rb') as file:
    y_val = pickle.load(file)
with open('sentence_matrices_test.pkl', 'rb') as file:
    sentence_matrices_test = pickle.load(file)
with open('tfidf_vectorizer_Ytest.pkl', 'rb') as file:
    y_test = pickle.load(file)


# In[61]:


tensor_train = torch.tensor(sentence_matrices_train, dtype=torch.float32)
labels_tensor_train = torch.tensor(y_train, dtype=torch.long)
print(tensor_train.shape)
print(labels_tensor_train)

# tensor_train = tensor_train.view(-1,1, -1)
# print("Reshaped input data shape:", tensor_train.shape)

tensor_test = torch.tensor(sentence_matrices_test, dtype=torch.float32)
labels_tensor_test = torch.tensor(y_test, dtype=torch.long)
# tensor_test = tensor_test.view(len(tensor_test),1, -1)
# print("Reshaped input data shape:", tensor_test.shape)


# In[62]:


tensor_val = torch.tensor(sentence_matrices_val, dtype=torch.float32)
labels_tensor_val = torch.tensor(y_val, dtype=torch.long)
# tensor_val = tensor_val.view(len(tensor_val),1, -1)
# print("Reshaped input data shape:", tensor_val.shape)


# ## LSTM

# ### Dataset and DataLoader

# In[63]:


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[64]:


dataset = CustomDataset(tensor_train, labels_tensor_train)
train_data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


# In[65]:


dataset = CustomDataset(tensor_test, labels_tensor_test)
test_data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


# In[66]:


dataset = CustomDataset(tensor_val, labels_tensor_val)
val_data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


# In[67]:


train_data_loader


# In[68]:


for x, y in train_data_loader:
    print(x.shape,y.shape)
    # print(y)
    break
for x, y in test_data_loader:
    print(x.shape,y.shape)
    # print(y)
    break


# ### Basic LSTM Network

# In[69]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[71]:


class LSTM(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim,64,num_layers=1,bidirectional=True,dropout=0.5,batch_first=True)
        self.fc = nn.Linear(128,2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x,(hidden_state,cell_state)=self.lstm(x)
        hidden=torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        x=self.fc(hidden)
        x=self.sigmoid(x)
        return x


# In[72]:


input_dim = 100
embed_dim = 100
model = LSTM(embed_dim).to(device)
# model_lstm = model_lstm.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


# In[73]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Print layer shapes
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")


# ### Basic Training Loop

# In[55]:


# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[74]:


train_losses = []
val_losses = []
epochs  = 5
for epoch in range(epochs):  # Number of epochs
    model.train()
    for inputs, labels in train_data_loader:
        optimizer.zero_grad()
        output = model(inputs)  # Transpose input for LSTM
        # labels_float = labels.unsqueeze(1).float()
        loss = criterion(output, labels)
        # loss = criterion(output, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        # val_loader = DataLoader(, batch_size=best_batch_size)
        val_losses_epoch = []
        for inputs, labels in val_data_loader:
            output = model(inputs)  # Transpose input for LSTM
            # labels_float = labels.unsqueeze(1).float()
            loss = criterion(output, labels)
            # loss = criterion(output, labels.unsqueeze(1))
            val_losses_epoch.append(loss.item())
        val_losses.append(np.mean(val_losses_epoch))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")



import matplotlib.pyplot as plt

plt.plot(range(epochs), train_losses, label='train')
plt.plot(range(epochs), val_losses, label='eval')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LSTM_plot.png')


correct = 0
total = 0
for x, y in train_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    # print(yp)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    # print(yp)
    # print(y)
    total += len(y)
    
print(f"Accuracy on train Data {(correct * 100 / total):.2f}")
correct = 0
total = 0
for x, y in val_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    # print(yp)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    # print(yp)
    # print(y)
    total += len(y)
    
print(f"Accuracy on val Data {(correct * 100 / total):.2f}")
correct = 0
total = 0
for x, y in test_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    # print(yp)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    # print(yp)
    # print(y)
    total += len(y)
    
print(f"Accuracy on test Data {(correct * 100 / total):.2f}")



# In[79]:


import pickle
with open('LSTM_Model.pkl', 'wb') as file:
    pickle.dump(model, file)



