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
Much of the architecture of the AlexNet is implemented to predict class labels
on the CIFAR10 dataset.

Example (RNN): Character Level prediction
-----------------------------------------
An LSTM implementation of character level prediction of Shakespaere's works and
the linux kernel (similar as XX) can be seen in XX.

Acknoledgements
---------------
I created this class mainly to learn about c++ and deep learning papers. I
leared a lot about the various implementations from caffee and tensorflow. I
also used a third party package to load MNIST and CIFAR10 data from XX.
