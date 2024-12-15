#!/usr/bin/env python
# coding: utf-8

# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pickle
# Load TF-IDF vectorizer from file
with open('tfidf_vectorizer_Xtrain.pkl', 'rb') as file:
    tfidf_vectorizer_train = pickle.load(file)
with open('tfidf_vectorizer_Xtest.pkl', 'rb') as file:
    tfidf_vectorizer_test = pickle.load(file)
with open('tfidf_vectorizer_Xval.pkl', 'rb') as file:
    tfidf_vectorizer_val = pickle.load(file)
with open('tfidf_vectorizer_Ytrain.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open('tfidf_vectorizer_Ytest.pkl', 'rb') as file:
    y_test = pickle.load(file)
with open('tfidf_vectorizer_Yval.pkl', 'rb') as file:
    y_val = pickle.load(file)
# tfidf_vectorizer_train = np.concatenate((tfidf_vectorizer_train.toarray(), tfidf_vectorizer_val.toarray()), axis=0)
# y_train = np.concatenate((y_train, y_val), axis=0)


# In[3]:


print(tfidf_vectorizer_train.shape)
tfidf_tensor_train = torch.tensor(tfidf_vectorizer_train.todense(), dtype=torch.float32)
labels_tensor_train = torch.tensor(y_train, dtype=torch.long)
tfidf_tensor_val = torch.tensor(tfidf_vectorizer_val.todense(), dtype=torch.float32)
labels_tensor_val = torch.tensor(y_val, dtype=torch.long)
tfidf_tensor_test = torch.tensor(tfidf_vectorizer_test.todense(), dtype=torch.float32)
labels_tensor_test = torch.tensor(y_test, dtype=torch.long)
# tfidf_tensor_Xtrain_Xval = torch.tensor(X_train_val_tfidf, dtype=torch.float32)
# labels_tensor_Xtrain_Xval = torch.tensor(y_train_val, dtype=torch.long)
print(tfidf_tensor_train.shape)
print(labels_tensor_train)
# labels_batch = labels_batch.unsqueeze(1)


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[5]:


class CustomDataset(Dataset):
    def __init__(self, tfidf, labels):
        self.tfidf = tfidf
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tfidf[idx], self.labels[idx]


# In[6]:


# # Create dataset and data loader
from torch.utils.data import ConcatDataset
dataset = CustomDataset(tfidf_tensor_train, labels_tensor_train)
train_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset = CustomDataset(tfidf_tensor_val, labels_tensor_val)
val_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

dataset = CustomDataset(tfidf_tensor_test, labels_tensor_test)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# dataset = CustomDataset(tfidf_tensor_Xtrain_Xval, labels_tensor_Xtrain_Xval)
# train_val_data_loader = DataLoader(dataset, batch_size=31, shuffle=True)


# In[7]:


# Initialize a counter
num_samples = 0

# Iterate over the combined data loader
for batch in train_data_loader:
    # Increment the counter by the batch size
    num_samples += batch[0].size(0)  # Assuming batch[0] contains the input data

# Print the total number of samples
print("Total number of samples in the combined data loader:", num_samples)


# In[8]:


for x, y in train_data_loader:
    print(x.shape,y)
    # print(y)
    break
for x, y in test_data_loader:
    print(x.shape,y.shape)
    # print(y)
    break


# In[9]:


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, 8)
        # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        out = self.fc3(out)
        # out = self.softmax(out)
        return out


# In[10]:


# Hyperparameters
input_size =  len(tfidf_tensor_train[0]) # Size of TF-IDF vectors
hidden_size = 4
output_size = 2  # Number of classes
num_layers = 2
dropout = 0.5
print(input_size)


# In[11]:


model = Network(input_size, hidden_size, output_size)
from torchsummary import summary
summary(model, (1, input_size))


# In[12]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[13]:


train_epoch_loss = []
eval_epoch_loss = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    curr_loss = 0
    total = 0
    for tfidf_batch, labels_batch in train_data_loader:
        # Forward pass
        outputs = model(tfidf_batch)
        loss = criterion(outputs, labels_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        curr_loss += loss.item()
        total += len(labels_batch)
    
    # Calculate average training loss for the epoch
    train_epoch_loss.append(curr_loss / total)
    
    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss[-1]:.4f}')

    curr_loss = 0
    total = 0
    for eval_x, eval_y in val_data_loader:
        optimizer.zero_grad()

        with torch.no_grad():
            y_pred = model(eval_x)

        loss = criterion(y_pred, eval_y)

        curr_loss += loss.item()
        total += len(eval_y)
    eval_epoch_loss.append(curr_loss / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss[-1]:.4f}, val Loss: {eval_epoch_loss[-1]:.4f}')


# In[14]:

import matplotlib.pyplot as plt

plt.plot(range(num_epochs), train_epoch_loss, label='train')
plt.plot(range(num_epochs), eval_epoch_loss, label='eval')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot as an image file
plt.savefig('DNN_plot.png')





# In[15]:


correct = 0
total = 0
for x, y in train_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on train Data {(correct * 100 / total):.2f}")
correct = 0
total = 0
for x, y in val_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on Validation Data {(correct * 100 / total):.2f}")
correct = 0
total = 0
for x, y in test_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")


# In[ ]:





# In[16]:


import optuna


# In[17]:


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NeuralNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
                # layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
                # layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# In[18]:


epochs = 10
input_size = len(tfidf_tensor_train[0])  # Assuming tfidf_tensor_train is defined

def objective(trial):
    # Define hyperparameters to tune
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "num_layers": trial.suggest_int("num_layers", 1, 1),
        "hidden_size": trial.suggest_int("hidden_size", 2, 10, log=True),
    }

    model = NeuralNetwork(input_size=input_size, hidden_size=params["hidden_size"], output_size=2, num_layers=params["num_layers"])
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(accuracy)
    return accuracy


# In[19]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the final model using the best hyperparameters
best_learning_rate = best_params["learning_rate"]
best_num_layers = best_params["num_layers"]
best_hidden_size = best_params["hidden_size"]

# final_model = NeuralNetwork(input_size=input_size, hidden_size=best_hidden_size, output_size=2, num_layers=best_num_layers)
# final_optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate)


# In[20]:


final_model = NeuralNetwork(input_size=input_size, hidden_size=best_hidden_size, output_size=2, num_layers=best_num_layers)
final_optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate)
train_epoch_loss = []
eval_epoch_loss = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    curr_loss = 0
    total = 0
    for tfidf_batch, labels_batch in train_data_loader:
        # Forward pass
        outputs = final_model(tfidf_batch)
        loss = criterion(outputs, labels_batch)

        # Backward pass and optimization
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()
        
        curr_loss += loss.item()
        total += len(labels_batch)
    
    # Calculate average training loss for the epoch
    train_epoch_loss.append(curr_loss / total)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss[-1]:.4f}')

    curr_loss = 0
    total = 0
    for eval_x, eval_y in val_data_loader:
        final_optimizer.zero_grad()

        with torch.no_grad():
            y_pred = final_model(eval_x)

        loss = criterion(y_pred, eval_y)

        curr_loss += loss.item()
        total += len(eval_y)
    eval_epoch_loss.append(curr_loss / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], val Loss: {loss.item():.4f}')


# In[21]:


import matplotlib.pyplot as plt

plt.plot(range(num_epochs), train_epoch_loss, label='train')
plt.plot(range(num_epochs), eval_epoch_loss, label='eval')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('DNN_plot.png')


# In[22]:


correct = 0
total = 0
for x, y in val_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = final_model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on val Data {(correct * 100 / total):.2f}")
correct = 0
total = 0
for x, y in test_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = final_model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")


# In[23]:


print(final_model.parameters)


# In[24]:


import pickle
with open('DNN_Model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

