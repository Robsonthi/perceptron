# perceptron
 Three data groups with three perceptrons
 
 Main file: perceptron.py

## Result

![image](https://github.com/user-attachments/assets/dbc4a108-e973-483f-8b4e-e524115ff32d)

## Perceptron
Perceptron is a classification approach, it’s a simple artificial neural network model that is used to learn binary classifiers. Based on a dot product (or matrix multiplication) and a threshold. With a set of data (inputs) with size $$n$$, you perceive a multiplication with a set of weights (W).

$$X=
\begin{bmatrix}
1 &
x_1 &
x_2 &
\cdots &
x_n
\end{bmatrix}$$

$$X$$ has a shape $$1 \times (n+1)$$,

$$W=
\begin{bmatrix}
w_0 &
w_1 &
w_2 &
\cdots &
w_n
\end{bmatrix}^T$$

$$W$$ has a shape $$(n+1) \times 1$$.

The classification is defined for:

$$X \cdot W = \begin{bmatrix}
1 &
x_1 &
x_2 &
\cdots &
x_n
\end{bmatrix}
\cdot
\begin{bmatrix}
w_0 \\
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
= w_0 +
\sum_{i=1}^{n} x_i \cdot w_i$$

When you have a database with patterns, and want to classify others related data, you must train the perceptron. The training is an iterative algorithm, whereupon the weights are adjusted according to the input data pattern.

The basic calculation is a dot product. The dot product is a measure of how closely two vectors align, in terms of the directions they point. However, the matrix multiplication of perceptron is related with a sample belongs on a hyperspace and the weights that represent a hyperplane. The hyperplane is a form of separate hyperspace.

The perceptron’s objective is the find the weights that separate the data from hyperspace. To update the weights, you use this:

$$W_{new} = 
\begin{bmatrix}
w_0 &
w_1 &
w_2 &
\cdots &
w_n
\end{bmatrix}_{old}^T +
\alpha \cdot (target-predict) \cdot
\begin{bmatrix}
1 &
x_1 &
x_2 &
\cdots &
x_n
\end{bmatrix}^T$$

The threshold is a classification form, if the result is positive, the data is on side of space, if the result is negative, the is on another side of space, separated for hyperplane.

The bias moves hyperplane out origin of space.
