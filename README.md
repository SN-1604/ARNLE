# ARNLE
Attentional Recurrent Network based on Language Embedding
## Run on
Python 3.6 <br>
Tensorflow 2.6.0 <br>
CUDA 11.8 <br>
CUDNN 8.9.1 <br>
Numpy 1.19.3 <br>
Pandas 0.23.0 <br>
Scikit-learn 0.23.2 <br>
Gensim 4.1.2 <br>
Logomaker 0.8 <br>

## Installation
To install environment：
```Python
pip install -r requirement.txt
```
The ELMo language model pre-trained with coronavirus ORF1ab, Spike, Envelop, Membrane and Nucleocapsid is available on Zenodo (DOI:10.5281/zenodo.8207208). The supervised Bi-LSTM classifier trained on coronavirus Spike protein can also be obtained on Zenodo (named as "model_supervised_attentional_recurrent_network"). <br>
Demo to run embedding of given sequence.fasta:
```Python
python embedding --file './data/sequence_train_S.fasta'
                  --input_type fasta
                  --model_path 'path_of_language_model'
                  --output './embedding/embedded_sequence_train_S.npy'
                  --max_length 264
```
Then a numpy type file of embeded vector sequences of input 'sequence_train_S.fasta' will be output to './embedding/embedded_sequence_train_S.npy'. (size [sequence_amount, 264, 1024])
# Usage
### Amino acid sequence embedding: 
```Python
python embedding --file 'sequence.fasta'
                  --input_type fasta
                  --model_path 'path_of_language_model'
                  --output 'output_file_of_embedding_data'
                  --batchsize(optional) 256(default)
                  --max_length max_length_of_sequence_data
```
### To train supervised Bi-LSTM model:
```Python
python supervised Bi-LSTM train.py --data_train 'embedded_train_data'
                                  --label_train 'label_file_of_train_data'
                                  --length_train 'length_file_of_train_data'
                                  --data_val 'embedded_validation_data'
                                  --label_val 'label_file_of_validation_data'
                                  --length_val 'length_file_of_validation_data'
                                  --writer_path 'path_to_write_train_log'
                                  --model_path 'path_to_save_trained_model'
                                  --epoch 10(default) --keepprob 0.8(default)
                                  --num_class 6(default)
                                  --hidden_size '256,128'(default)
                                  --lr 1e-3(default)
                                  --max_length 264(default)
```
### To predict virus host tropism:
```Python
python predict.py --model_path 'path_of_trained_supervised_Bi-LSTM_model'
                  --data 'embedded_data_to_predict'
                  --file 'sequence_file_to_predict'
                  --out_path 'output_path_of_predicting_result'
                  --num_class 6(default)
                  --batchsize 256(default)
                  --n_split 6(default)
```

### Perform Post-hoc Bayesian explanation:
```Python
python bayes.py --data_frame 'summary_data_framework'
                --data_adapt 'output_tropic_data_framework'
                --data_nonadapt 'output_nontropic_data_framework'
```

### Analyze top amino acid site influencing virus host tropism:
```Python
python logo_plot_rm_x.py --data_adapt 'tropic_data_framework'
                        --data_nonadapt 'nontropic_data_framework'
                        --sorted_diff 'output_sorted_difference'
                        --out_path 'logo_plot_output_path'
```

### Calculate combination mutation bayesian probabilties difference on host tropism:
```Python
python combination_mutant_statistic.py --data_frame 'summary_data_framework'
                                      --out_dic 'output_difference_dictionary'
                                      --out_excel 'output_difference_excel'
```

### CTSI calculation
```Python
python CTSI_over_time.py --bayes_monthly 'path_of_monthly bayesian_difference'
                        --output 'output_file'
```
