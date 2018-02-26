# Hello MNIST
A learning-oriented implementation of an artificial neural network in C++ that performs basic OCR, 
inspired by this wonderful [YouTube series created by 3Blue1Brown](http://3b1b.co/neural-networks)
and Michael Neilsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). 
The code can be found [here](https://github.com/mnielsen/neural-networks-and-deep-learning). Very much like Nielsen's 
code, this project chooses to forgo some optimizations for the sake of keeping the code readable for people getting 
into ML. 

## To-Do
- [ ] Write back-propagation algorithm
- [ ] Write parser for the data set to convert to arrays of floats (or steal one already written)
- [ ] Write stochastic gradient descent code.
- [ ] Write testing code. 
 
## So... What does the code do?
The neural network trains on the from a modified NIST dataset (hence
the name). You can then feed it a new hand labelled character in hopefully it will properly identify it.

## Why would anyone care about _this_ repo? 
The main reasons I bothered to code this are:
* To get more comfy with C++ and CMake. 
* To have a convenient codebase I can use for learning more C++ stuff
* To have a starting point for learning to implement CUDA kernels. 
* To gather my own experimental data on how NN architecture, training algorithms, and other hyperparameters affect the 
performance of the neural net on the MNIST dataset. 

For people breaking into the ML/Deep Learning scene with a strong C++ background, this could be a nice aid to 
Neilsen's Python code. 

On top of that, this neural network could easily be re-purposed for another dataset or to solve another problem. If 
you really like my code for whatever reason, you could also hypothetically use this as the basis for building some 
fancier  stuff. 

## How the testing works (WIP)

One of the quirky parts of this neural net is its testing methodology. I have a test setup where I attempt to classify 
vectors whose magnitude is greater than one. This simple methodology allows for pretty rigorous testing that's a little 
simpler than OCR, making it easier to track errors in convergence, floating point arithmetic, etc. 