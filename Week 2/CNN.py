# %%
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
import sys


# %%
import pickle

with open('tfidf_vectorizer_Ytrain.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open('tfidf_vectorizer_Ytest.pkl', 'rb') as file:
    y_test = pickle.load(file)
with open('tfidf_vectorizer_Yval.pkl', 'rb') as file:
    y_val = pickle.load(file)
# tfidf_vectorizer_train = np.concatenate((tfidf_vectorizer_train.toarray(), tfidf_vectorizer_val.toarray()), axis=0)
# y_train = np.concatenate((y_train, y_val), axis=0)
# Load the saved embeddings
X_train_uncased_embeddings = torch.load(f'X_train_{sys.argv[1]}.pt')
X_val_uncased_embeddings = torch.load(f'X_val_{sys.argv[1]}.pt')
X_test_uncased_embeddings = torch.load(f'X_test_{sys.argv[1]}.pt')

# %%
# print(X_train_uncased_embeddings.shape)
# tfidf_tensor_train = torch.tensor(X_train_uncased_embeddings, dtype=torch.float32)
labels_tensor_train = torch.tensor(y_train, dtype=torch.long)
# tfidf_tensor_val = torch.tensor(X_val_uncased_embeddings.todense(), dtype=torch.float32)
labels_tensor_val = torch.tensor(y_val, dtype=torch.long)
# tfidf_tensor_test = torch.tensor(X_test_uncased_embeddings.todense(), dtype=torch.float32)
labels_tensor_test = torch.tensor(y_test, dtype=torch.long)
# tfidf_tensor_Xtrain_Xval = torch.tensor(X_train_val_tfidf, dtype=torch.float32)
# labels_tensor_Xtrain_Xval = torch.tensor(y_train_val, dtype=torch.long)
# print(tfidf_tensor_train.shape)
# print(labels_tensor_train)
# labels_batch = labels_batch.unsqueeze(1)

# %%
import torch
import optuna
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
# X_train_uncased_embeddings = torch.load('X_train_bert.pt')

# X_test_uncased_embeddings = torch.load('X_test_bert.pt')

# X_val_uncased_embeddings = torch.load('X_val_bert.pt')

def correct_form_data(X, y):
    # Assuming your list is named 'sparse_matrices'
    dense_matrices = []
    for sparse_matrix in X:
        tmp = torch.tensor(sparse_matrix, dtype = torch.float32)
        dense_matrices.append(tmp)
    dense_matrices = torch.stack(dense_matrices)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    dataset = TensorDataset(dense_matrices, y)
    return dataset
train_dataset = correct_form_data(X_train_uncased_embeddings,y_train)
val_dataset = correct_form_data(X_val_uncased_embeddings,y_val)

def objective(trial):
    params = {
    "num_filters" : trial.suggest_int('num_filters', 32, 128),
    "dropout_rate" : trial.suggest_uniform('dropout_rate', 0.2, 0.5),
    "kernel_sizes" : [trial.suggest_int(f'kernel_size_{i}', 1, 3) for i in range(3)],
    "batch_size" : trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    # Create DataLoader for training dataset
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    # Create DataLoader for validation dataset
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    model = CNN_Text(train_dataset[0][0].shape[0], params["num_filters"], params["kernel_sizes"], params["dropout_rate"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):  # Number of epochs
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_losses = []
        for inputs, labels in val_loader:
            output = model(inputs)
            loss = criterion(output, labels.unsqueeze(1))
            val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
    best_trial_checkpoint_path = f"./checkpoints/best_checkpoint_CNN_trial_{trial.number+1}.pt"
    torch.save({'state':model.state_dict(),
                        'params':params}, best_trial_checkpoint_path)
    # checkpoint_path = f"./checkpoints/best_checkpoint_CNN_trial_{trial.number+1}.pt"
    # torch.save({'model': model.state_dict(),
    #                     'batch_size' : batch_size,
    #                     'num_filters' : num_filters,
    #                     'kernel_sizes': kernel_sizes,
    #                     'dropout_rate': dropout_rate,
    #                     "trial" : trial}, checkpoint_path)
    return avg_val_loss

class CNN_Text(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, dropout_rate):
        super(CNN_Text, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=ks, padding=1)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        x = x.unsqueeze(2)  # Conv1d expects (batch, channel, seq_len)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

val_losses = []
train_losses = []

# Train the model with best hyperparameters
best_model = CNN_Text(train_dataset[0][0].shape[0], best_params['num_filters'], [best_params[f'kernel_size_{i}'] for i in range(1)], best_params['dropout_rate'])
criterion = nn.BCELoss()
optimizer = optim.Adam(best_model.parameters())

# Create DataLoader with best batch size for training
best_batch_size = best_params['batch_size']
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)

for epoch in range(10):  # Number of epochs
    best_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = best_model(inputs)
        loss = criterion(output, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    best_model.eval()
    with torch.no_grad():
        val_losses_epoch = []
        val_loader = DataLoader(val_dataset, batch_size=best_batch_size)
        for inputs, labels in val_loader:
            output = best_model(inputs)
            loss = criterion(output, labels.unsqueeze(1))
            val_losses_epoch.append(loss.item())
        val_losses.append(np.mean(val_losses_epoch))

    print(f"Epoch {epoch+1}/{10}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")

# Test the final model
y_pred = []
y_true = []
best_model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = best_model(inputs)
        predicted = (outputs >= 0.5).float()
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.squeeze().numpy())
accuracy = sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
# print("Validation accuracy of the final model:", accuracy)

# Generate classification report
# print(classification_report(y_true, y_pred))
y_pred = []
y_true = []
test_dataset = correct_form_data(X_test_uncased_embeddings,y_test)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size)
y_pred = []
y_true = []
best_model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_model(inputs)
        predicted = (outputs >= 0.5).float()
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.squeeze().numpy())
accuracy = sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
print("---------------CNN Model Accuracy:---------------")
print("Testing accuracy of the final model:", accuracy)

# Generate classification report
print(classification_report(y_true, y_pred))
# dbfile = open('test_loader_cnn.pkl', 'wb')
# pickle.dump(test_loader, dbfile)
# dbfile.close()
# dbfile = open('CNN.pkl', 'wb')
# pickle.dump(best_model, dbfile)
# dbfile.close()
