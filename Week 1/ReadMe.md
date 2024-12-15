# DL Assignment-Week 1


**Overview:**

This repository contains scripts and files for a machine learning pipeline designed to classify social media posts as either fake or real. The pipeline consists of several tasks, including dataset preparation, preprocessing of social media text, obtaining vector representations, training ML classification models, and evaluating model performance.

**Tasks:**

1. **Prepare dataset:** The dataset is split into training, validation, and test sets using scikit-learn's `train_test_split` function. The split ratio is 80/10/10, and the splits are saved as CSV files.

2. **Preprocessing Social Media Post:** The preprocessing step handles non-textual elements in social media posts, such as emojis, URLs, and hashtags, to preserve the integrity of the text data.

3. **Obtaining vector representations:** Textual data is encoded into TF-IDF vectors to prepare it for input into machine learning models.

4. **Training DL classification models:** Various DL binary classification models are trained and tuned, including DNN ,CNN, LSTM model

5. **Evaluating Machine Learning Models:** The performance of the tuned machine learning models is evaluated using metrics such as confusion matrix, classification accuracy, F1-score, precision, and recall on the test split obtained in Task-1. The final best set of hyperparameters specific to each model is reported.

**Instructions to Run:**


1. Execute the Makefile file to run the entire machine learning pipeline. The input file to the model will be CL-II-MisinformationData - Sheet1.csv.

2. Follow the prompts and instructions displayed during the execution of the script. Ensure that all required files and scripts are present in the same directory before running the main script.

**Files and Scripts:**

- `PreprocessAndVectorize.py`: 
- `CNN.py`, `DNN.py`, `LSTM.py`: Scripts to train and tune specific DL classification models.
- `RunEval.py`: Script to evaluate the trained models using the test split.

**Notes:**
- **NOTE THE FUNCTION TAKES TIME TO RUN**
- The pipeline execution may take some time depending on the size of the dataset and complexity of the models.
- For any questions or clarifications, please refer to the documentation or contact the repository owner.


**Instructions for custom test file:**

1. Excecute Evauate_Test_Custom.py file and provide filename.csv
