# Readme

LSTM recurrent neural network is one of deep learning techniques.
We hope to analyse the sentiment of IMDB reviews using LSTM classifier.
Online tutorial and original python codes are from:
http://deeplearning.net/tutorial/lstm.html

## notes
`Optimizer: adadelta`, which is an adaptive learning rate method.
So, initial learning rate (0.001) has no effect and is not needed.
(AdaDelta is preferable over SGD and RMSProp.)

`dim_proj`, LSTM number of hidden units. varied from 128 to 1024 in this project.

`maxlen`, max sequance length. If a sequence longer than this, it will get ignored.
    only sequences shorter than `maxlen` words would be processed (patched with 0's and feed to LSTM).

`epoch`, `for eidx in range(max_epochs)` iterate to get parameter matricies optimized.
    will stop when no improvement could make.


max sentence length  | accuracy
--- | ---
70  | 
100 | 
150 | 
200 | 

hidden dimension  | accuracy
--- | ---
128 | 
256 | 
512 | 
1024  | 

No. epochs | accuracy
--- | ---
1 | 
10  | 
20  | 

## end
last edit 04/19/2016
