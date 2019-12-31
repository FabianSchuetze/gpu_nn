A Deeplearning framework in C++

Summary
-------
The deeplearning framework can be used to train neural network on a CPU or a
GPU. The feedforward and backwards function of each layer have an GPU and CPU
implementation. Memort for for forward and backward pass is automatically
managed and allocated between the devices as needed. 

Strenghts
------------
1. Simple GPU and CPU implementation. 
2. Modularity: The framework can be extended easily. Similar layers can be
   included by inheriting the design of
   XX. Simlarly, further optimizers can be added by inheriting from XX, metrics
   by inheriting from XX, weight initializers are defined with referece to XX
   and further loss function can be added by inheriting from XX.
3. Speed: The implementation speed is regularly on par with Keras  implementations called from Python.

Examples
--------
####A deep neural network for MNIST
----------------------------------------
The standard MNIST example is developed in [test/mnist.cpp](test/mnist.cpp). Running this file, prints the following information to `stdout`
```shell
Layer 0: Input, output size: 784 
Layer 1: Dense, output size: 1024 
Layer 2: Relu, output size: 1024 
Layer 3: Dropout, output size: 1024 
Layer 4: Dense, output size: 1024 
Layer 5: Relu, output size: 1024 
Layer 6: Dropout, output size: 1024 
Layer 7: Dense, output size: 10 
Layer 8: Softmax, output size: 10 
features, target60000, 60000
train loss after 500 iter: 1.84551
train loss after 1000 iter: 1.0458
train loss after 1500 iter: 0.738511
after iter 0 the loss is 0.403217, in 21420 milliseconds
fraction miassclassified : 0.102167 and number missclassified 613
train loss after 2000 iter: 0.59197
train loss after 2500 iter: 0.534108
train loss after 3000 iter: 0.492661
after iter 1 the loss is 0.288599, in 21421 milliseconds
fraction miassclassified : 0.0813333 and number missclassified 488
train loss after 3500 iter: 0.462315
train loss after 4000 iter: 0.431044
train loss after 4500 iter: 0.41157
after iter 2 the loss is 0.247715, in 21153 milliseconds
fraction miassclassified : 0.0718333 and number missclassified 431
```

#### (CNN): CIFAR10 with AlexNet
-----------------------------
Much of the architecture of the [AlexNet] (https://code.google.com/archive/p/cuda-convnet/) is implemented [here](test/cifar/cifar10.cpp) to predict class labels on the CIFAR10 dataset. When running this model, the output for the first epoch is:
```shell
Layer 0: Input, output size: 3 32 32 
Layer 1: Im2ColLayer, output size: 1024 
Layer 2: Convolution, output size: 64 32 32 
Layer 3: Pooling, output size: 64 16 16 
Layer 4: Im2ColLayer, output size: 256 
Layer 5: Convolution, output size: 64 16 16 
Layer 6: Pooling, output size: 64 8 8 
Layer 7: Im2ColLayer, output size: 64 
Layer 8: Convolution, output size: 128 8 8 
Layer 9: Pooling, output size: 128 4 4 
Layer 10: Dense, output size: 128 
Layer 11: Dropout, output size: 128 
Layer 12: Relu, output size: 128 
Layer 13: Dense, output size: 10 
Layer 14: Dropout, output size: 10 
Layer 15: Softmax, output size: 10 
features, target50000, 50000
train loss after 500 iter: 2.18132
train loss after 1000 iter: 2.07094
after iter 0 the loss is 1.74208, in 180749 milliseconds
fraction miassclassified : 0.5264 and number missclassified 2632
```
To speed up the training, the model adds `Im2Col` layers before a convolutional layer.


#### (RNN): Character Level modeling:
----------------------------------------
A replication of the LSTM model by Karphaty, Johnson and Fei-Fei to model Shakespeare's works can be seen in [test/rnn](test/rnn/rnn.cpp). As the model trains, it displays sampled dialogues, such as:

>All:
Now she be well, for her father, sigh or to thee for her,
And with a life and be a time and so so stolen us,
And he shall see the best heart of my best and grieved
And when you shall be well the news to keep thee on:
And so we will and the father to me all.
>
>TRANIO:
Her with a suitors to the father.
>
>PETRUCHIO:
For I say, sith the suitors do be your motion
And the first shall be thine first: and once not redeems,
For our profit and I will plead you to the word.

While the text obviously wasn't written by Shakespeare I'm particularly amazed by how consistently the model references objects (father, or suitor) accross speaekers. 


Acknoledgements
---------------
I created this class mainly to learn about c++ and deep learning papers. I leared a lot about the from reading the code of caffe and tensorflow. I also used a third party package to load MNIST and CIFAR10 data from XX.

To Do
-------
1. After spending several hours debugging the derivatives of the forward
   propagation functions, I realize how handy automatic differentation is. It
   would be nice to implement this here
2. I would also like to know how to implement a distributed training, such as
   [Hogwild](https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent) (which is often used in models with spares inputs such as word embeddings) or [Distbelief](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
3. At the moment, the code can only backpropagate from one head, a more
   flexible implementation would be nice.

Tests
-----
All the layers and optimizers have both GPU and CPU functions. The equivalence
of the results is checked with dedicated tests. Similary, I test that the GPU
implementation is never slower than the CPU version.


