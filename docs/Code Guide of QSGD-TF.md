# Code Guide of QSGD-TF

This is a code guide of our implementation of QSGD-TF. Based on [Horovod](https://github.com/uber/horovod), we introduce QSGD method to quantize the gradients and obviously reduce the communication time cost in multi-node training. 

There are two main changes we did to Horovod:
1. Communication 
    In every batch, each node gets training data and calculates gradients, which are stored in tensors. Originally, Horovod uses `ncclAllReduce` (if supported) or `MPI_AllReduce` to aggregate tensors from all nodes with full precision. Whereas in QSGD-TF, we removed the `ncclAllReduce` and `MPI_AllReduce`, and redesigned the communication pattern as two rounds:
    - In first round, we get tensors to be exchanged from Horovod directly and take them as one big vector. Let's say we have `n` nodes in cluster. Then we partition this vector into `n` chunks, so that each node just aggregates one chunk. They quantize these chunks to low precision and send them to corresponding nodes with `MPI_Isend`. At the same time, each node `MPI_Irecv` one compressed chunk from all other nodes and conduct dequantization. For example, every node sends first chunk to first node, and second chunk to second node.
    - After first round, each node has one different aggregated chunk. So the second round is to send their own aggregated chunk to all other nodes. Similarly, each node should quantize the data before sending and dequantize it after receiving.  

2. Quantization
In this implementation we quantized the gradients number from 32-bit `float` to 8-bit `unsigned char` to communicate. The quantization algorithm can be described as follows (it is implemented in function `GPUQuantizeValue` and `GPUDequantizeValue`):
    - Imagine there is a list `L` of 32-bits numbers we want quantize. We first find out the `maximum` and `minimum` number of list `L` (using `GPUFindMaxAndMin` function). Then we partition the interval between `maximum` and `minimum` into `255` parts (because we use 8 bits to represent each number). Then each number in list `L` falls into one specific part between two integer numbers.
    - Our aim is to use one integer number between 0 to 255 to represent each number in list `L`. So we apply stochastic rounding to choose one integer number between these two.
    - To dequantize the received list, we also need the `maximum` and `minimum` number of this list, so these two number should be send in full precision. Then we can get the `unit` length of this list as : `(max - min) / 255`. For each received number `L[i]`, the dequantized value should be : `min + L[i] * unit`.
