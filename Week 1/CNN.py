#!/usr/bin/env python
# coding: utf-8

# In[61]:


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



# In[62]:


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


# In[74]:


tensor_train = torch.tensor(sentence_matrices_train, dtype=torch.float32)
labels_tensor_train = torch.tensor(y_train, dtype=torch.long)
size = len(tensor_train[0])
print(labels_tensor_train)

# tensor_train = tensor_train.view(-1,1, -1)
# print("Reshaped input data shape:", tensor_train.shape)

tensor_test = torch.tensor(sentence_matrices_test, dtype=torch.float32)
labels_tensor_test = torch.tensor(y_test, dtype=torch.long)
# tensor_test = tensor_test.view(len(tensor_test),1, -1)
# print("Reshaped input data shape:", tensor_test.shape)



# In[64]:


tensor_val = torch.tensor(sentence_matrices_val, dtype=torch.float32)
labels_tensor_val = torch.tensor(y_val, dtype=torch.long)
# tensor_val = tensor_val.view(len(tensor_val),1, -1)
# print("Reshaped input data shape:", tensor_val.shape)


# ## 2-Dimentional CNN

# ### Dataset and DataLoader

# In[65]:


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[126]:


dataset = CustomDataset(tensor_train, labels_tensor_train)
train_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[127]:


dataset = CustomDataset(tensor_test, labels_tensor_test)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[128]:


dataset = CustomDataset(tensor_val, labels_tensor_val)
val_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[129]:


train_data_loader


# In[130]:


for x, y in train_data_loader:
    print(x.shape,y)
    # print(y)
    break
for x, y in test_data_loader:
    print(x.shape,y.shape)
    # print(y)
    break


# ### Basic Network

# In[131]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[171]:


import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_channel=1, output_dim=2):
        super(Network, self).__init__()
        self.conv_1 = nn.Conv2d(input_channel, 32, (7, 7), stride=5)
        self.activation_1 = nn.ReLU()
        # self.batch_norm_1 = nn.BatchNorm2d(32)
        # self.pool1 = nn.MaxPool2d((2, 2), stride=1)
        
        self.conv_2 = nn.Conv2d(32,64, (5, 5), stride=5)
        self.activation_2 = nn.ReLU()
        # self.batch_norm_2 = nn.BatchNorm2d(32)
        # self.pool2 = nn.MaxPool2d((2, 2), stride=1)
    
        # You had an incorrect size for batch_norm_2, corrected it to 32
        self.calculate_conv_sizes()
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear_3 = nn.Linear(64* self.new_height * self.new_width, output_dim)
        self.activation_3 = nn.Sigmoid()
        
    def calculate_conv_sizes(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, size, 100)
            conv1_out = self.conv_1(dummy_input)
            conv1_out = self.activation_1(conv1_out)
            # conv1_out = self.batch_norm_1(conv1_out)
            # conv1_out = self.pool1(conv1_out)
            
            conv2_out = self.conv_2(conv1_out)
            conv2_out = self.activation_2(conv2_out)
            # conv2_out = self.batch_norm_2(conv2_out)
            # conv2_out = self.pool2(conv2_out)
            
            # Calculate the new dimensions
            self.new_height, self.new_width = conv2_out.size(2), conv2_out.size(3)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.activation_1(out)
        # out = self.batch_norm_1(out)
        # out = self.pool1(out)
      
        out = self.conv_2(out)
        out = self.activation_2(out)
        # out = self.batch_norm_2(out)
        # out = self.pool2(out)
        
        out = self.flatten(out)
        # out = self.dropout(out)
        out = self.linear_3(out)
        out = self.activation_3(out)
        return out


# In[172]:


network = Network().to(device)


# In[173]:


from torchsummary import summary

# Assuming your model is named model and your training data tensor is named tensor_train
input_length = tensor_train.size(2)
print(input_length)
summary(network,  (1, size, 100))


# ### Basic Training Loop

# In[174]:


criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(network.parameters(), lr = 0.0006)
epochs = 20


# In[175]:


train_epoch_loss = []
eval_epoch_loss = []
for epoch in tqdm(range(epochs)):
    curr_loss = 0
    total = 0
    for train_x, train_y in train_data_loader:
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optim.zero_grad()
        y_pred = network(train_x)
        loss = criterion(y_pred, train_y)
        loss.backward()
        optim.step()
        curr_loss += loss.item()
        total += len(train_y)
    train_epoch_loss.append(curr_loss / total)
    # print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_epoch_loss[-1]:.4f}')
    curr_loss = 0
    total = 0
    for eval_x, eval_y in val_data_loader:
        eval_x = eval_x.to(device)
        eval_y = eval_y.to(device)
        optim.zero_grad()

        with torch.no_grad():
            y_pred = network(eval_x)

        loss = criterion(y_pred, eval_y)
        curr_loss += loss.item()
        total += len(train_y)
    eval_epoch_loss.append(curr_loss / total)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_epoch_loss[-1]:.4f}, val Loss: {eval_epoch_loss[-1]:.4f}')


# In[176]:


import matplotlib.pyplot as plt

plt.plot(range(epochs), train_epoch_loss, label='train')
plt.plot(range(epochs), eval_epoch_loss, label='eval')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('CNN_plot.png')


# ### Testing on Test dataset

# In[190]:


correct = 0
total = 0
for x, y in train_data_loader:
    x = x.to(device)
    with torch.no_grad():
        yp = network(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
#     print(yp)
#     print(y)
print(f"Accuracy on train Data {(correct * 100 / total):.2f}")


correct = 0
total = 0
for x, y in val_data_loader:
    x = x.to(device)
    with torch.no_grad():
        yp = network(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
#     print(yp)
#     print(y)
print(f"Accuracy on val Data {(correct * 100 / total):.2f}")

correct = 0
total = 0
for x, y in test_data_loader:
    x = x.to(device)
    with torch.no_grad():
        yp = network(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
#     print(yp)
#     print(y)
print(f"Accuracy on test Data {(correct * 100 / total):.2f}")
model_accuracy = correct * 100 / total


with open('CNN_Model.pkl', 'wb') as file:
    pickle.dump(network, file)



