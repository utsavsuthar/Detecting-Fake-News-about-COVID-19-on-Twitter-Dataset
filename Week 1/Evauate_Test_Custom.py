import subprocess
import os
import PreprocessAndVectorize 
file_name = input("Enter the CSV File Name:")


current_directory = os.path.dirname(os.path.abspath(__file__))

# preprocess_process = subprocess.Popen(["python3", os.path.join(current_directory, 'PreprocessAndVectorize.py')])
# preprocess_process.wait()
PreprocessAndVectorize.preprocess(file_name)
# Run the RunEval.py scripts for DNN, CNN, and LSTM
models = ['DNN', 'CNN', 'LSTM']
subprocess.Popen(["python3", os.path.join(current_directory, 'RunEval.py'), 'DNN','tfidf_matrix_new_test.pkl', 'y_new_test.pkl'])
    # eval_process.wait()
subprocess.Popen(["python3", os.path.join(current_directory, 'RunEval.py'), 'CNN','sentence_matrices_new_test.pkl', 'y_new_test.pkl'])
    # eval_process.wait()
subprocess.Popen(["python3", os.path.join(current_directory, 'RunEval.py'), 'LSTM','sentence_matrices_new_test.pkl', 'y_new_test.pkl'])
    # eval_process.wait()
print("All subprocesses completed.")
