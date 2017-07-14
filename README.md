# Why LSTM over RNN?

### Drawbacks of RNN :

**1. Slow training process:**
- In case of long sequences, the gradients (values calculated to tune the network) computed during the training (backpropagation) either vanish (multiplication of many 0 < values < 1)
or explode ( multiplication of many large values ).

**2. Forgetful nature of RNN:**
- Vanishing gradient problem manifests itself in case of RNN, making RNN Aamir Khan from the Bollywood block-buster [Ghajini](http://www.imdb.com/title/tt1166100/).
- **Gradient of the loss function** decays exponentially with time.
- RNNs have trouble in remembering values of past inputs after more than 10 timesteps approx.
- Dire need to have some control over how past inputs are preserved.


**How does RNN remember sequence information?** <br>
All RNNs have feedback loops in the recurrent layer. This lets them maintain information in 'memory' over time.

<b>How does LSTM remember what RNN can not because of <i>vanishing gradients of the loss function</i>?</b>

LSTMs are special..:P
- Comes with **memory cell** that can maintain information in memory for long periods of time.
- A <u>set of gates</u> is used to control the flow of information.
	- Set of gates help in preserving the inputs from the past. How?
	- By controlling:<br>
		i) how much of information enters the memory<br>
		ii) how much of information is forgotten<br>
		iii) and thus passing (information that entered the memory - information that is forgotten) as output.<br>

To innately and intuitively understand how information of the past is passed across the network in RNN/LSTM, look at this [image](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)<br>
![alt-text](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/hidden_layer_recurrence.png)


### LSTM in a Nut-shell

**Workflow**
Input => Cell/Memory Unit => Output/State Update

**Convention followed below:**
- x is same as x(t)
- h is same as h(t-1)
- c is same as c(t-1)


**Activation functions represented as :** f, g<br>
**Activation functions used are :** sigmoid, tanh<br>

**Basic facts about the activation function used:**<br>
**Sigmoid :** sigmoid(z) is always between 0 and 1<br>
**tanh :** tanh(z) is always between -1 and 1<br>

**Dimensional Analysis:**<br>
f(W,b,x,h) => <br>
	    x is a vector of shape (1,p)<br>
	    h is a vector of shape (1,q)<br>
	    W is a matrix of shape (p+q,m) => Since the actual input is x vector concatenated with h vector and thus of dimension (1, p+q)<br>
	    b is a matrix of shape (1,m)<br>
	
**Input :**<br>
xt concatenated with h(t-1) => Xt<br>
		+<br>
cell state, c(t-1) => Only LSTMs maintain a cell state (which also helps overcoming vanishing and exploding gradient problem). RNNs do not have any concept of Cell state.<br>

	  

**Cell :**<br>
 Gates => Gate : In general, function of (W,b,x,h) where the function is also referred to as activation.<br>
       	- Input Gate <br>
		- f(W1,b1,x,h) <br>
	- Output Gate <br>
		- f(W2,b2,x,h) <br>
	- Forget Gate (not present in Vanilla LSTM) <br>
		- f(W3,b3,x,h) <br>
       Input Transform <br>
		- g(W4,b4,x,h)<br>
	

**Output :**<br>
cell state, c(t) = Forget Gate * c + input gate * input transform<br> 
(where * means elementwise multiplication between vectors, also called as Hadamard Product.)<br>
	+<br>
hidden state, h(t) = Output Gate * g(c(t)) = f(W2,b2,x,h) * g(c(t))<br>

Cliched but True, *A picture is worth a thousand words.*
![alt text](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/mathematics_of_lstm.svg)

<u>Note</u> : The above mentioned LSTM in a Nut-Shell is explained (here)[https://apaszke.github.io/lstm-explained.html].
















	








