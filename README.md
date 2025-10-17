# JS-DigitRecognition
A pure JavaScript code to train/run a model to recognize hand written digits, no library of both external and build-in is used.

## About
This is a hello world practise after I learnt the very basic of ANN(Watched a couple video about ANN on youtube). As mentioned above it's pure JS, tested on Node.js, should be working on Web JS as well, but never tested. Trained using the MNIST hand written digit data base

## Arch
28x28 Input layer, 128 Node Hidden, 128 Node Hidden, 10 node for output
Backpropagation + Sigmoid + Squared error + Stochastic Gradient Decent (The code kept the option for Relu but it's not used by default nor any test runs)

## Performance
trainning accruecy and test accurecy of around 93.5% using the MNIST hand wirtten digit data base

Output of run.js for training data:
```
Sample 60000/60000 | elapsed: 8.4s | ETA: 0.0s | accuracy: 94.78%
Final Accuracy: 94.78%
```
Output of run.js for testing data
```
Sample 10000/10000 | elapsed: 1.4s | ETA: 0.0s | accuracy: 93.02%
Final Accuracy: 93.02%
```
