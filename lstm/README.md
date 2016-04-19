# Readme

LSTM recurrent neural network is one of deep learning techniques.
We hope to analyse the sentiment of IMDB reviews using LSTM classifier.
Online tutorial and original python codes are from:
http://deeplearning.net/tutorial/lstm.html

The best accuracy is 86.1% with 1024 hidden units.

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

keep `maxlen=70`, vary `dim_proj`

hidden dimension  | accuracy
--- | ---
128 | 78.6%
256 | 
512 | 82.8%
1024  | 83.8%

Below table shows that usually iteration stops within 20 epochs because of no accuracy improvement.

No. epochs | accuracy
--- | ---
1 | 50%
10  |   82.8%
20  |   82.8%
30  |   80%
40  |   79.2%
50  |   79.7%
60  |   80.4%


## end
last edit 04/19/2016
