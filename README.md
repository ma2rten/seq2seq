# Sequence to Sequence

This is a Python implementation of the Sequence to Sequence model described in the paper:

    Sequence to Sequence Learning with Neural Networks by Sutskever et.al, NIPS 2014.

I hope this implementation helps others to understand this algorithm.  My main focus in this implementation was readability and understandability. For this reason I did not implement mini-batches and the only dependency is numpy, as a result this code does make use of GPUs.

## Toy examples

For the reasons mentioned above this implementation is too slow to apply to any large scale real-world problem. Instead, I included two toy problems which demonstrate the learning ability of the model.

###  Memorization

It is a common trick to ensure the correctness of a machine learning algorithm to feed a small dataset to a new implementation to make sure the training error will decrease to zero in absense of regularization.

### Counting

This toy tasks is inspired by one of the toy tasks in the original LSTM paper. We feed a random sequence of a's and b's to the model. We expect model to return the same number of a's ignoring the b's. This requires the LSTM to keep and internal "counter" in it's hidden state of the number of a's that it sees.

## Notes

Unlike the original description of the algorithm, this implementation does not use called Peephole connections. There seems to be some consensus in the RNN community that these do not improve accuracy, but increase computational complexity.

The cell state and hidden initialized to zero for each new sequence (an alternative approach would be to learn the initial value)

Gradient clipping is implemented by taking the L2 norm of all model parameters' gradients combined (as opposed to layer by layer)
