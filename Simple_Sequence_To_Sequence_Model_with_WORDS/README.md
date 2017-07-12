# LSTM for Language Modelling :
- It is a Simple Sequence to Sequence Model.
- Given 3 input words (not characters) generate next n words to form a sentence/story.
- Using frequency of each word to generate word to index dictionary (highest frequency means least index).
- It is **NOT** a Language Translation ( Encoder-Decoder ) Model.
- Link for [visual understanding](https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537).

### How to use [LSTM-Tensorflow.ipynb](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/Simple_Sequence_To_Sequence_Model_with_WORDS/LSTM-Tensorflow.ipynb)
- The containing repository comes along with [*input_text.txt*](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/Simple_Sequence_To_Sequence_Model_with_WORDS/input_text.txt).<br>
- *input_text.txt* is a regular file containing a list of sentences.<br>
- *LSTM-Tensorflow.ipynb* by default runs taking input from *input_text.txt*.<br>
- In order to run the code with custom file, change the variable named **training_file**'s value to your <file-name>.<br>
![alt-text](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/Simple_Sequence_To_Sequence_Model_with_WORDS/change_input_file.gif)
