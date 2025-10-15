# JS-DigitRecognition
A pure JavaScript code to train/run a model to recognize hand written digits, no library of both external and build-in is used.

## About
This is a hello world practise after I learnt the very basic of ANN(Watched a couple video about ANN on youtube). As mentioned above it's pure JS, tested on Node.js, should be working on Web JS as well, but never tested. Trained using the MNIST hand written digit data base

## Arch
28x28 Input layer, 128 Node Hidden, 128 Node Hidden, 10 node for outpu
Backpropagation + Sigmoid + Squared error + Stochastic Gradient Decent (The code kept the option for Relu but it's not used by default nor any test runs)

## Performance
trainning accruecy and test accurecy of around 95% using the MNIST hand wirtten digit data base
