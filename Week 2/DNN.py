# %%
# from google.colab import drive
# drive.mount('/content/drive')


# %%
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
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings("ignore")
import sys

import pickle
X_test_embeddings = torch.load(f'X_test_{sys.argv[1]}.pt')

with open('tfidf_vectorizer_Ytrain.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open('tfidf_vectorizer_Ytest.pkl', 'rb') as file:
    y_test = pickle.load(file)
with open('tfidf_vectorizer_Yval.pkl', 'rb') as file:
    y_val = pickle.load(file)

X_train_embeddings = torch.load(f'X_train_{sys.argv[1]}.pt')
X_val_embeddings = torch.load(f'X_val_{sys.argv[1]}.pt')

labels_tensor_train = torch.tensor(y_train, dtype=torch.long)

labels_tensor_val = torch.tensor(y_val, dtype=torch.long)

labels_tensor_test = torch.tensor(y_test, dtype=torch.long)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)


# %%
class CustomDataset(Dataset):
    def __init__(self, tfidf, labels):
        self.tfidf = tfidf
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tfidf[idx], self.labels[idx]

# %%
# # Create dataset and data loader
from torch.utils.data import ConcatDataset
dataset = CustomDataset(X_train_embeddings, labels_tensor_train)
train_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset = CustomDataset(X_val_embeddings, labels_tensor_val)
val_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

dataset = CustomDataset(X_test_embeddings, labels_tensor_test)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# dataset = CustomDataset(tfidf_tensor_Xtrain_Xval, labels_tensor_Xtrain_Xval)
# train_val_data_loader = DataLoader(dataset, batch_size=31, shuffle=True)


class Network(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, output_size):
        super(Network, self).__init__()
 
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        # out = self.softmax(out)
        return out


# %%
# Hyperparameters
input_size =  len(X_train_embeddings[0]) # Size of TF-IDF vectors
# hidden_size1 = 512
# hidden_size2 = 256
# hidden_size3 = 128
output_size = 2  # Number of classes
# num_layers = 2
# dropout = 0.5
# print(input_size)

import optuna
epochs = 10
criterion = nn.CrossEntropyLoss()

def objective(trial):
    # Define hyperparameters to tune
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
        # "num_layers": trial.suggest_int("num_layers", 1, 3),
        "hidden_size1": trial.suggest_int("hidden_size1", 256, 512, log=True),
         "hidden_size2": trial.suggest_int("hidden_size2", 128, 256, log=True),
         "hidden_size3": trial.suggest_int("hidden_size3", 32, 128, log=True),
    }

    model = Network(input_size=input_size, hidden_size1=params["hidden_size1"],hidden_size2=params["hidden_size2"],hidden_size3=params["hidden_size3"], output_size=2 )
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
    # print(accuracy)
    best_trial_checkpoint_path = f"./checkpoints/best_checkpoint_DNN_trial_{trial.number+1}.pt"
    torch.save({'state':model.state_dict(),
                        'params':params}, best_trial_checkpoint_path)
    return accuracy


# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the final model using the best hyperparameters
best_learning_rate = best_params["learning_rate"]
# best_num_layers = best_params["num_layers"]
best_hidden_size1 = best_params["hidden_size1"]
best_hidden_size2 = best_params["hidden_size2"]
best_hidden_size3 = best_params["hidden_size3"]
# final_model = NeuralNetwork(input_size=input_size, hidden_size=best_hidden_size, output_size=2, num_layers=best_num_layers)
# final_optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate)

# %%
final_model = Network(input_size=input_size, hidden_size1=best_hidden_size1, hidden_size2=best_hidden_size2,hidden_size3=best_hidden_size3,output_size=2)
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

# %%
import matplotlib.pyplot as plt

plt.plot(range(num_epochs), train_epoch_loss, label='train')
plt.plot(range(num_epochs), eval_epoch_loss, label='eval')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('DNN_plot.png')


# %%
correct = 0
total = 0
for x, y in train_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = final_model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
print(f"Accuracy on Train Data {(correct * 100 / total):.2f}")
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
y_pred = []
y_test = []
for x, y in test_data_loader:
    # x = x.to(device)
    with torch.no_grad():
        yp = final_model(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    y_pred += yp
    y_test += y
    correct += (yp == y).sum()
    total += len(y)
# print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")
print("---------------DNN Model Accuracy:---------------")
print(f"Accuracy on test Data {(correct * 100 / total):.2f}")
print(classification_report(y_test, y_pred))