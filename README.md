# Transformer

Implementation of the Transformer model from the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762).

### Model Architecture

![The full transformer architecture ](fig/transformer.png)

#### Encoder and Decoder Stacks:
Encoder: 
*  $N=6$ identical layers.
* Each layer has 2 sub-layers:
	* A multi-head self-attention mechanism.
	* Position-wise fully connected feed-forward network.
* Each of the 2 sub-layers:
	* A residual connection.
	* Layer normalization.
	$LayerNorm(x + Sublayer(x))$
* All sub-layers, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$.

Decoder:
* $N=6$ identical layers.
* In additional to the 2 sub-layers, the decoder inserts a third sub-layer which performs multi-head attention over the output of the encoder stack.
* The self-attention sub-layer in the decoder stack is modified to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

#### Attention
Mapping a query and a set of key-value pair to an output.
* The query, keys, values and output are all vectors.
* The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Scaled Dot-Product Attention

Multi-head Attention

#### Position-wise Feed-Forward Networks

#### Embeddings and Softmax

#### Positional Encoding

### Training

#### Optimizer
Adam optimizer with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$

#### Regularization
Residual Dropout
Label Smoothing
### References
* [Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/01/attention.html)
