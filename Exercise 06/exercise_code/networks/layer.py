import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a mini-batch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(d_out, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    :param d_out: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,

    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dw = np.reshape(x, (x.shape[0], -1)).T.dot(d_out)
    dw = np.reshape(dw, w.shape)

    db = np.sum(d_out, axis=0, keepdims=False)

    dx = d_out.dot(w.T)
    dx = np.reshape(dx, x.shape)

    return dx, dw, db


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = 1 / (1 + np.exp(-x))
        cache = outputs
        return outputs, cache

    def backward(self, d_out, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = d_out * cache * (1 - cache)
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################

        outputs = np.maximum(x, 0)
        cache = outputs

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, d_out, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################

        dx = cache.copy()
        dx[dx >= 0] = 1
        dx[dx < 0] = 0
        dx = d_out * dx

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class LeakyRelu:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of LeakyRelu activation function          #
        ########################################################################

        outputs = x.copy()
        outputs[outputs < 0] *= self.slope

        cache = outputs

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, d_out, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of LeakyRelu activation function         #
        ########################################################################

        dx = cache.copy()
        dx[dx >= 0] = 1
        dx[dx < 0] = self.slope
        dx = d_out * dx

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Tanh activation function               #
        ########################################################################

        exp_x = np.exp(x)
        exp_neg_x = np.exp(-x)
        outputs = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        cache = outputs

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, d_out, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Tanh activation function              #
        ########################################################################

        dx = d_out * (1 - cache * cache)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx
