from datasets import load_dataset
import pandas as pd
import sys
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import torch


# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# Custom Dataset class for handling tokenization
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['tweet']
        label = 1 if self.data.iloc[idx]['label'] == 'real' else 0
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'labels': label}



def training(model_name):
   
    # Create train and validation datasets
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)

    # Define the model
    # model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification

    # Define the training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        output_dir="./results",
        overwrite_output_dir=True,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    evaluation_result = trainer.evaluate()

    print("Evaluation result:", evaluation_result)

    model_save_path="automodel_"+model_type+".pth"
    torch.save(model.state_dict(), model_save_path)

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

model_type='bert-base-uncased'
model_t1=["covid-twitter-bert","bert-base-uncased","bert-base-cased","twhin-bert-base","SocBERT-base"]
import sys
modelname = sys.argv[1]
if modelname == 'bert-base-cased':
    bert_model = 'bert-base-cased'
    bert_model2 = 'bert-base-cased'
elif modelname == 'bert-base-uncased':
    bert_model = 'bert-base-uncased'
    bert_model2 = 'bert-base-uncased'
elif modelname == 'covid-twitter-bert':
    bert_model = 'digitalepidemiologylab/covid-twitter-bert'
    bert_model2 = 'covid-twitter-bert'
elif modelname == 'twhin-bert-base':
    bert_model = 'Twitter/twhin-bert-base'
    bert_model2 = 'twhin-bert-base'
elif modelname == 'socbert':
    bert_model = 'sarkerlab/SocBERT-base'
    bert_model2 = 'socbert'
# if "covid-twitter-bert" in model_type:
#     model_type="digitalepidemiologylab/covid-twitter-bert"
# elif "twhin-bert-base" in model_type:
#     model_type="Twitter/twhin-bert-base"
# elif "SocBERT-base" in model_type:
#     model_type="sarkerlab/SocBERT-base"


tokenizer = AutoTokenizer.from_pretrained(bert_model, token='ENTER YOUR API KEY ')
training(model_type)

# python3 autoModel.py bert-base-uncased 
