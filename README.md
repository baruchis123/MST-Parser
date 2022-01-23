# MST Parser
### In this section of the NLP course in the Computer Science program at Hebrew University, we learned about different models designed to predict correct dependencies (how words of a sentence interact with one another) between words of a sentence. One of these models is the MST Parser which predicts edges of a graph between the different words of a sentence and then runs one of the many algorithms used for finding a Maximum Spanning Tree (inference of the model) over said graph to find the most likely predictions.
## Implementation
Our model is designed as a discriminative log-linear model, with a score function, which given a pair of nodes on a dependency graph (a pair of words), and a given feature function <img src="https://render.githubusercontent.com/render/math?math=\Phi"> (in our implementation, it is a very simple function, will be elaborated upon shortly) predicts an edge to the graph as seen here:

<div align="center"><img src="https://render.githubusercontent.com/render/math?math=score_{\theta}(v_{1},v_{2} | x_{1},...,x_{n}) = \theta^{t}\cdot\Phi(v_{1},v_{2},x_{1},...,x_{n})" height="50" width="500"></div>

The model will be implemented in the class
```python
class MST_Parser
```

### Feature Function
The feature function in our program will encode two simple features:
1. A score of one if the two nodes match two words, and a score of zero if not - essentialy our implementation will return a vector of the dimension <img src="https://render.githubusercontent.com/render/math?math=|vocabulary|^{2}"> with one coordinate as one and the rest zeroes. The feature is implemented in the method:
```python
def _feature_function_word_bigram(self, word1, word2)
```

2. A score of one if the POS (parts of speech - verb, noun, adjective etc.) of the two nodes match a certain two POS - same as the previous encoding, our implementation will return a vector of size <img src="https://render.githubusercontent.com/render/math?math=|vocabulary|^{2}"> with one coordinate as one and the rest zeroes. The feature is implemented in the method:
```python
def _feature_function_POS_bigram(self, word1, word2)
```

### Training
Training of our model will be done using stochastic gradient descent and the Perceptron Average algorthim (switches the need to find in every iteration the expectation of all the spanning trees and just corrects the model according to the maximum spanning tree), where in each iteration of training, we will find the MST and correct our models parameters (<img src="https://render.githubusercontent.com/render/math?math=\theta">) accordingly - training of the algorithm will be done using the following function:
```python
def train(self, n_iterations: int, batch_size: int, sentences)
```

### Testing
After finding the best parameters through the training stage, testing will be done on the test set using the parameters found through the training process.
