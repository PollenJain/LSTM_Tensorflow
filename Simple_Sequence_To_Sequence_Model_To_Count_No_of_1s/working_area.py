''' Using Map '''


print(map(int, '123')) # Returns a map object. In python 3x, its a generator.
print(list(map(int, '123'))) # [1, 2, 3]
print(list('123')) # [ '1', '2', '3' ]


l = ['123', '456', '789']
print([ list(string) for string in l]) # [ ['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'] ]
print([ list(map(int, string)) for string in l ]) # [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]
train_input = [ list(map(int, string)) for string in l ]

words = [ 'my', 'name', 'is', 'Anthony', 'Gonsalves' ]
print( [ list(string) for string in words ] ) # [ [ 'm', 'y' ],
					      #   [ 'n', 'a', 'm', 'e' ],
					      #   [ 'i', 's' ]
					      #   [ 'A', 'n', 't', 'h', 'o', 'n', 'y' ]
					      #   [ 'G', 'o', 'n', 's', 'a', 'l', 'v', 'e', 's']
					      # ]



''' Using np.split '''

import numpy as np

# input has to be a numpy array. Can't be a Python list.
x = np.split(np.array([1,2]), 2) # For a change, it returns a Python List and not numpy array
print(x) #[ array([1]), array([2]) ]
# print(x.shape) # 'list' object has no attribute 'shape'
print(x[0].shape) # (1,)

print(np.split(np.array([1,2,3]), 3)) # [array([1]), array([2]), array([3])] => we want each element to be a 2-dimensional array

print(np.split(np.array([[1,2,3] ] ), 3, axis=1)) # [array([1]), array([2]), array([3])]
# Now, we are fine :) :)

''' Using np.expand_dims '''

i = np.expand_dims([1,2,3], axis=0)
print(i) #[ [1,2,3] ]
j = np.expand_dims([1,2,3], axis=1)
print(j) # [ [1], [2], [3] ]


three = np.array( [ [ [1,2] ], [ [2,3] ], [ [4,5] ] ] ) # (3, 1, 2)
print(three.shape)
print(np.sum(three, axis=2)) # [ [3], [5], [9] ]

three_col = np.array( [ [ [1], [2]  ], [ [2], [3] ], [ [4], [5] ] ] )
print(three_col.shape) # (3, 1, 2)
print(np.sum(three_col, axis=1)) # [ [3] , [5], [9] ]


''' Using np.transpose(array, axes) same as using tf.transpose(array, perm)'''

out = np.array([ [ [1,2,11,12], [10,20,30,40] ], [ [3,4,13,14], [20,40,60,80] ], [ [5,6,7,8], [33,44,55,66] ] ]) # (3, 2, 4) => 3 is at index 0, 2 is at index 1, 4 is at index 2
print(out.shape)
print(out)
out_t_1 = np.transpose(out, axes=(1, 0, 2)) # I want shape to be (2, 3, 4). We want (3, 2, 4) or (index 0, index 1, index 2) to become (2, 3, 4) or (index 1, index 0, index 2)

print(out_t_1.shape) # (2, 3, 4)
print(out_t_1) 
# [ [
#	[ 1  2 11 12]
# 	[ 3  4 13 14]
#  	[ 5  6  7  8]
#   ]

# [
#	[10 20 30 40]
# 	[20 40 60 80]
# 	[33 44 55 66]
# ] ]



''' Using tf.transpose '''

import tensorflow as tf

out_tensor = tf.transpose(out, perm=(1,0,2))
sess = tf.Session()
res = sess.run(out_tensor)

print(res.shape) # (2, 3, 4)
print(res)

# [ [
#	[ 1  2 11 12]
# 	[ 3  4 13 14]
#  	[ 5  6  7  8]
#   ]

# [
#	[10 20 30 40]
# 	[20 40 60 80]
# 	[33 44 55 66]
# ] ]


''' Using tf.gather '''

g = tf.gather(res, 0) # returns 1st 2-dimensional numpy array in res
rg = sess.run(g)
print(rg.shape) # (3, 4)
print(rg)
#[	[ 1  2 11 12]
# 	[ 3  4 13 14]
#	[ 5  6  7  8]
#]


g1 = tf.gather(res, 1) # returns 2nd 2-dimensional numpy array in res
rg1 = sess.run(g)
print(rg1.shape) # (3, 4)
print(rg1)
# [ [10 20 30 40]
#   [20 40 60 80]
#   [33 44 55 66]
# ]


# To return the last 2-dimensional numpy array in res
last_two_dimensional_numpy_array = tf.gather(res, res.shape[0]-1) # res.shape : (2, 3, 4) => res.shape[0] is 2 => 2-1 is 1 => tf.gather(res, 1) => returns the 2-dimensional array in res at index 1
r = sess.run(last_two_dimensional_numpy_array)
print(r.shape)
print(r)
# [	[10 20 30 40]
# 	[20 40 60 80]
# 	[33 44 55 66]
#]


''' Using RNN specific Tensorflow functions '''

num_hidden = 3
x = tf.placeholder(tf.float32, [1, 2, 1])
cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
output1, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
output = tf.transpose(output1, perm=(1, 0, 2))
last = tf.gather(output, output.get_shape()[0]-1)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # We need this even though we don't have any tf.Variable defined.
l = np.array([ [ [1], [0] ] ])

print(l.shape) # (1, 2, 1)
out,s, out_transposed,last_output_transposed = sess.run((output1,state,output,last), feed_dict={ x:l }) 
print(out.shape) # (1, 2, 3) => When num_hidden = 3 => (1, 2, num_hidden) 
print(out) 
#[	[
#		[ 0.13414799  0.08370781  0.14346786]
#  		[ 0.08594143  0.06918873  0.0906116 ]
#	]
#]
#print(s.shape) #Its an object of type LSTMStateTuple and not numpy array. Therefore, it has no attribute 'shape'.
print(s) # When num_hidden=3 , both c and h are of shape (1,3) or (1, num_hidden)
#LSTMStateTuple(c=array([[-0.16679749, -0.1860403 ,  0.01415037]], dtype=float32), h=array([[-0.08188692, -0.09595101,  0.00704593]], dtype=float32))
c = s[0]
print(c.shape) # (1, 3) => (1, num_hidden)
h = s[1]
print(h.shape) # (1, 3) => (1, num_hidden)
print(out_transposed) # (2, 1, 3)
print(out_transposed.shape)
#[	[
#		[ 0.13414799  0.08370781  0.14346786]
#	]
#	[
#  		[ 0.08594143  0.06918873  0.0906116 ]
#	]
#]
print(last_output_transposed.shape) # (1,3)
print(last_output_transposed) 
print(np.sum(last_output_transposed)) # Sum need not be 1. These are not probablities.
#	[
#  		[ 0.08594143  0.06918873  0.0906116 ]
#	]

