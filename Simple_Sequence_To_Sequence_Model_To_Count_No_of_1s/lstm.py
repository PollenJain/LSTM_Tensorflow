''' http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/ '''

from random import shuffle

''' Generating Dataset '''

n = 2
n_str = str(n)
# Note: replace 2 by str(n) in '{0:02b}'
train_data = [ '{0:02b}'.format(i) for i in range(2**n) ] 
print(train_data) # [ '00', '01', '10', '11' ] 

''' Problem is to find/count number of 1s in a given string '''

# So for '00', no of 1s is 0
# and for '11', no of 1s is 2
# In 2**2, maximum number of 1s is 2
# Our output will be a 1 hot encoded vector. So, if there are no ones, we will represent it as [1 0 0], at index 0 (no of ones is represented by the index) we have a 1
# if there are 1 one, then it will be represented as [0 1 0]
# Maximum number of ones we can have is 2, therefore it will be [0 0 1]
# So to represent all possible cases of 2**2 we need an array with index [0, 1, 2] in the output for one-hot encoding
# If we had to represent all possible cases of 2**n we would need an array with index [0, 1, 2, ..., n] in the output for one-hot encoding => size is (n+1) 

# (n+1) can be seen as no of classes

# so '00' belong to class [1 0 0]
#    '01' belong to class [0 1 0]
#    '10' belong to class [0 1 0]
#    '11' belong to class [0 0 1]

''' Shuffle the dataset '''

# shuffle(train_data)  # Maybe not :P

train_input = [list(map(int, string)) for string in train_data] # [ [0, 0], [0, 1], [1, 0], [1, 1] ]

print(train_input)

''' How does Tensorflow wants the data for RNN? '''

# It says that if '00' can be seen as a word, and all the words in the train_data have same number of characters then,
# considering word '00',
# Have a 2 dimensional array, 
# with no of rows = no of characters in that word
# with no of columns per row = length of 1 character = 1 :P
# Therefore, '00' becomes => [ [0], [0] ] => shape (2, 1)

# Now since train_data has 4 such words, therefore train data will have
# 4 such (2,1) numpy arrays
# Therefore, shape of train data to be passed as input should be [4, 2, 1] :) :)

# It is also referred as, [batch_size, sequence_length, input_dimension]

''' Preparing training data for Tensorflow '''

import numpy as np

preparing_data = []
for i in train_input:
	preparing_data.append(np.expand_dims(i, axis=1)) # Expanding dimension along the columns (axis=1)

print(preparing_data)
''' [array([[0],
       [0]]), array([[0],
       [1]]), array([[1],
       [0]]), array([[1],
       [1]])] '''

print(preparing_data[0].shape) #(2, 1)


print(np.array(preparing_data).shape) # (4, 2, 1)

tensorflow_training_input = np.array(preparing_data) # (4, 2, 1)


''' Generating the training output data using tensorflow_training_input'''

# Training Output for '00' should be [1 0 0]
# '01' [0 1 0]
# '10' [0 1 0]
# '11' [0 0 1]

#Training output : [ [1 0 0], [0 1 0], [0 1 0], [0 0 1] ]

training_output = []
s = np.sum(tensorflow_training_input, axis=1) # [ [0], [1], [1], [2] ]

for i in range(2**n):
	one_hot_encoded_i = np.zeros((n+1,))
	ix = s[i,0]
	one_hot_encoded_i[ix] = 1
	training_output.append(one_hot_encoded_i)

print(np.array(training_output)) # [array([ 1.,  0.,  0.]), array([ 0.,  1.,  0.]), array([ 0.,  1.,  0.]), array([ 0.,  0.,  1.])]

tensorflow_training_output = np.array(training_output)

print(tensorflow_training_output.shape) # (4, 3) => Tensorflow RNN takes output as a 2-dimensional tensor


''' Dividing tensorflow_training_input into training (70%) and testing (30%) '''

training_size =  int(0.7*(2**n))
testing_size = (2**n) - training_size

print(training_size) # 2
print(testing_size) # 2

tensorflow_train_input = tensorflow_training_input[0:training_size]
print(tensorflow_train_input)
tensorflow_train_output = tensorflow_training_output[0:training_size]
print(tensorflow_train_output)


tensorflow_test_input = tensorflow_training_input[training_size:] # Using training_size itself instead of test_size 
print(tensorflow_test_input)
tensorflow_test_output = tensorflow_training_output[training_size:]
print(tensorflow_test_output)


''' Tensorflow Model '''

import tensorflow as tf



x = tf.placeholder(tf.float32, [None, n, 1]) # [Batch_size, sequence_length, input_dimension]
y = tf.placeholder(tf.float32, [None, n+1]) # one-hot encoded output


''' RNN Model '''
num_hidden = 24 # number of units in the LSTM Cell, hyperparameter

cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True) # returns a Tensorflow object

# Dimension of output1: (?, 2, 24) i.e., (?, n, num_hidden)
# Dimension of state: (?, 24) i.e., (?, num_hidden)

output1, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)


# Dimension of output: (2, ?, 24) i.e., (n, ?, num_hidden) => explained in working_area.py
output = tf.transpose(output1, [1, 0, 2]) # Switches batch size with sequence size

# Dimension of last: (?, 24) i.e., (?, num_hidden) => explained in working_area.py
last = tf.gather(output, int(output.get_shape()[0]) - 1) # we are interested only in the last output and none of the intermediate output.


# y.get_shape() is (?, 3) i.e., (?, n+1) since one-hot encoded vector is of length (n+1) ie, 0 to n
weight = tf.Variable(tf.truncated_normal( [ num_hidden, int(y.get_shape()[1]) ] ) ) 
bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]] ) )

# Dimensional Analysis: Note: Here, the one-hot encoded labels are of length (n+1)
# 	last : (?, 24) or (?, num_hidden)
# 	weight : (24, 3) or ( num_hidden, n+1)
# 	mul(last, weight) : (?, 3) or (?, n+1)
#	bias : (3,) or (n+1, )


prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)


cross_entropy = -tf.reduce_sum(y* tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))


optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

''' Calculating the error on test data '''

mistakes = tf.not_equal(tf.argmax(y, 1), tf.argmax(prediction,1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

''' Execution of the Graph '''

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 1000
no_of_batches = int(len(tensorflow_train_input)/batch_size)
epoch = 5000 # max_iterations

for i in range(epoch):
	start = 0
	for j in range(no_of_batches):
		inp = tensorflow_train_input[start:start+batch_size]
		output = tensorflow_train_output[start:start+batch_size]
		start = start + batch_size
		sess.run(minimize, {x: inp, y: output})
	print("Epoch", i)




''' Applying the learnt model on test data '''
incorrect = sess.run(error, {x:tensorflow_test_input, y:tensorflow_test_output})
print("error", 100*incorrect)


#print(sess.run(cell)) # cell is a tensorflow object and not a Tensor specifically.
#print(output)
#print(state)
#print(output1)
#print(last) 
#print(y.get_shape())

print(sess.run(prediction, {x: [ [ [1], [0] ] ] } )) # n = 2
sess.close()











