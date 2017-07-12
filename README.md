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

How does LSTM remember what RNN can not because of **vanishing gradients of the loss function**?

LSTMs are special..:P
- Comes with **memory cell** that can maintain information in memory for long periods of time.
- A <u>set of gates</u> is used to control the flow of information.
	- Set of gates help in preserving the inputs from the past. How?
	- By controlling:<br>
		i) how much of information enters the memory<br>
		ii) how much of information is forgotten<br>
		iii) and thus passing (information that entered the memory - information that 			    is forgotten) as output.<br>

In order to innately and intuitively understand how information of the past can be well understood from this [image](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)<br>
![alt-text](https://github.com/PollenJain/LSTM_Tensorflow/blob/master/hidden_layer_recurrence.png)







	








