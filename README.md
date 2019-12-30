A Deeplearning framework in C++

Summary
-------
The deeplearning framework can be used to train neural network on a CPU or a
GPU. The feedforward and backwards function of each layer have an GPU and CPU
implementation. Memort for for forward and backward pass is automatically
managed and allocated between the devices as needed. 

Strenghts
--------
1. Simple GPU and CPU implementation. 
2. Modularity: The framework can be extended easily. Similar layers can be
   included by inheriting the design of
   XX. Simlarly, further optimizers can be added by inheriting from XX, metrics
   by inheriting from XX, weight initializers are defined with referece to XX
   and further loss function can be added by inheriting from XX.
3. Speed: The implementation speed is regularly on par with Keras
   implementation called from Python.

Examples
--------

Example: A deep neural network for MNIST
----------------------------------------
The standard MNIST example is developed in the file XXX. Running this file,
prints the following information to XX


Example: (CNN) CIFAR10 with AlexNet
-----------------------------
Much of the architecture of the [AlexNet]
(https://code.google.com/archive/p/cuda-convnet/)
is implemented [here](test/cifar/cifar10.cpp) to predict class labels on the CIFAR10 dataset 


Example (RNN): Character Level modeling:
----------------------------------------
A replication of the LSTM model by Karphaty, Johnson and Fei-Fei to model 
Shakespeare's works can be seen in [test/rnn](test/rnn/rnn.cpp). 
As the model trains, it displays sampled 
dialogues, such as:
>All:
>Sir, thou wilt the scence is a sight:
>A imperion to the daughter than all the world?
>
>TRANIO:
>Have you me all the strength of the will we
>leave me to be her face to the deeds of his.


Acknoledgements
---------------
I created this class mainly to learn about c++ and deep learning papers. I
leared a lot about the various implementations from caffe and tensorflow. I
also used a third party package to load MNIST and CIFAR10 data from XX.

To Do
-----
I plan to work on a the following:
1. After spending several hours debugging the derivatives of the forward
   propagation functions, I realize how handy automatic differentation is. It
   would be nice to implement this here
2. I would also like to know how to implement a distributed training, such as
   [Hogwild](https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent) (which is often used in models with spares inputs such as word embeddings) or [Distbelief](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
3. At the moment, the code can only backpropagate from one head, a more
   flexible implementation would be nice.

