# ARNLE
Attentional Recurrent Network based on Language Embedding
## Run on
Python 3.6 \<br>
Tensorflow 2.5.0 \<br>
Numpy 1.19.3 \<br>
Pandas 0.23.0 \<br>
Scikit-learn 0.23.2 \<br>
Gensim 4.1.2 \<br>

# Usage
## Amino acid sequence embedding: 
Python embedding --file 'sequence.fasta' --input_type fasta --model_path 'path_of_language_model' --output 'output_file_of_embedding_data' --batchsize(optional) 256(default) --max_length max_length_of_sequence_data
## To train supervised Bi-LSTM model:
Python supervised Bi-LSTM train.py --data_train 'embedded_train_data' --label_train 'label_file_of_train_data' --length_train 'length_file_of_train_data' --data_val 'embedded_validation_data' --label_val 'label_file_of_validation_data' --length_val 'length_file_of_validation_data' --writer_path 'path_to_write_train_log' --model_path 'path_to_save_trained_model' --epoch 10(default) --keepprob 0.8(default) --num_class 6(default) --hidden_size '256,128'(default) --lr 1e-3(default) --max_length 264(default)
## To predict virus host tropism:
Python predict.py --model_path 'path_of_trained_supervised_Bi-LSTM_model' --data 'embedded_data_to_predict' --file 'sequence_file_to_predict' --out_path 'output_path_of_predicting_result' --num_class 6(default) --batchsize 256(default) --n_split 6(default)
