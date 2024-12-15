#!/bin/bash
python3 PreprocessAndVectorize.py
python3 DNN.py
python3 CNN.py
python3 LSTM.py
python3 RunEval.py DNN tfidf_vectorizer_Xtest.pkl tfidf_vectorizer_Ytest.pkl
python3 RunEval.py CNN sentence_matrices_test.pkl tfidf_vectorizer_Ytest.pkl
python3 RunEval.py LSTM sentence_matrices_test.pkl tfidf_vectorizer_Ytest.pkl