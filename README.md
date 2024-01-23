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
* ELMoformanylangs (https://github.com/berkay-onder/ELMoForManyLangs) is neccessary to run the embedding process. <br>
* The ELMo language model pre-trained with coronavirus ORF1ab, Spike, Envelop, Membrane and Nucleocapsid is available on Zenodo (DOI:10.5281/zenodo.8207208).
* The parameters of supervised Bi-LSTM classifier trained on coronavirus Spike protein can be obtained in the path 'model_supervised_attentional_recurrent_network'. <br>
# Usage
### Amino acid sequence embedding: 
```Python
python embedding.py --file 'sequence.fasta'
                  --input_type fasta
                  --model_path 'path_of_language_model'
                  --output 'output_file_of_embedding_data'
                  --batchsize(optional) 256(default)
                  --max_length max_length_of_sequence_data
                  --split(optinal) 6(default)
```
* Demo to run embedding of given sequence.fasta:
```Python
python embedding.py --file './data/sequence_train_S.fasta'
                  --input_type fasta
                  --model_path 'path_of_language_model'
                  --output './embedding/embedded_sequence_train_S.npy'
                  --max_length 264
```
Then a numpy type file of embeded vector sequences of input 'sequence_train_S.fasta' will be output to './embedding/embedded_sequence_train_S.npy'. (size [sequence_amount, 264, 1024])
### To train supervised Bi-LSTM classifier:
```Python
python supervised Bi-LSTM train.py --data_train 'embedded_train_data'
                                  --label_train 'label_file_of_train_data'
                                  --length_train 'length_file_of_train_data'
                                  --data_val 'embedded_validation_data'
                                  --label_val 'label_file_of_validation_data'
                                  --length_val 'length_file_of_validation_data'
                                  --writer_path 'path_to_write_train_log'
                                  --model_path 'path_to_save_trained_model'
                                  --epoch 10(default)
                                  --keepprob 0.8(default)
                                  --num_class 6(default)
                                  --hidden_size '256,128'(default)
                                  --lr 1e-3(default)
                                  --max_length 264(default)
```
* Demo to train a supervised Bi-LSTM classifier:
* Assuming that train and validation sequences have been embedded with 'embedding.py' and output to './embedding/embedded_sequence_train_S.npy' and './embedding/embedded_sequence_val_S.npy'. <br>
* Before training, it's neccessary to prepare files of sequence length and label of input sequences. <br>
* The length file should contain length of each sequence with line break '\n' separated. <br>
* The label file should contain host species label (human, bat, carnivora, artiodactyla, swine, rodentia) with line break '\n' separated. The program will automatically transfer the string label to integar label 0-5.
* Then the supervised Bi-LSTM classifier can be trained with:
```Python
python supervised Bi-LSTM train.py --data_train './embedding/embedded_sequence_train_S.npy'
                                  --label_train 'label_train.txt'
                                  --length_train 'length_train.txt'
                                  --data_val './embedding/embedded_sequence_val_S.npy'
                                  --label_val 'label_val.txt'
                                  --length_val 'length_val.txt'
                                  --writer_path 'path_to_write_train_log'
                                  --model_path 'path_to_save_trained_model'
```
The parameter '--writer_path' is the path to write training log of the classifier which can be load by TensorBoard to observe loss curves and variable state. <br>
The trained model will be writen in '--model_path' with model checkpoint of latest 5 epochs. So the model can be loaded by 'tf.train.import_meta_graph()'.
### To predict virus host tropism:
```Python
python predict.py --model_path 'path_of_trained_supervised_Bi-LSTM_classifier'
                  --data 'embedded_data_to_predict'
                  --file 'sequence_file_to_predict'
                  --out_path 'output_path_of_predicting_result'
                  --num_class 6(default)
                  --batchsize 256(default)
                  --n_split 6(default)
```
* The user can predict coronavirus host tropism with trained Bi-LSTM classifier we provided through parameter '--model_path'. <br>
* To predict host tropism, the origin sequence fasta is neccessary for program to automatically extract sequence length. <br>
* Then the host tropism probabilities numpy array (.npy) will be generated to '--out_path'.
### Perform Post-hoc Bayesian explanation:
```Python
python bayes.py --data_frame 'summary_data_framework'
                --data_adapt 'output_tropic_data_framework'
                --data_nonadapt 'output_nontropic_data_framework'
```
* The input '--data_frame' should be a pandas framework as:
  | id | country | data | site 1 | site 2 | ... | site N | Prob_PR | Prob_CH | Prob_CA | Prob_AR | Prob_SU | Prob_RD |
  | :--: | :-----: | :--: | :----: | :----:| :---: | :----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
  | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
* The program will generate two .csv files containing Bayesian probabilities of specific site and specific amino acid in tropic and non-tropic sequences.
### Analyze top amino acid site influencing virus host tropism:
```Python
python logo_plot_rm_x.py --data_adapt 'tropic_data_framework'
                        --data_nonadapt 'nontropic_data_framework'
                        --sorted_diff 'output_sorted_difference'
                        --out_path 'logo_plot_output_path'
```
* With input of tropic and non-tropic data framework output by 'bayes.py', the script will produce a new csv of sorted amino acid site influencing virus host tropism through parameter '--sorted_diff'.
* Two .png graph of tropic and non-tropic amino acid distribution in the top 20 sites will be produced through parameter '--out_path'.
* According to the insert of three amino acids in 213 site, all the site number over 213 of sequences without this insert mutation should be subtracted by 3 to align with the real site.
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
