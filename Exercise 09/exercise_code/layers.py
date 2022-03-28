import numpy as np

       
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from mini-batch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each time step we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        x_minus_mean = x - sample_mean
        sq = x_minus_mean ** 2
        var = 1.0 / N * np.sum(sq, axis=0)
        std = np.sqrt(var + eps)
        ivar = 1.0 / std
        x_norm = x_minus_mean * ivar
        gamma_x = gamma * x_norm
        out = gamma_x + beta
        running_var = momentum * running_var + (1 - momentum) * var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean

        cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, std, var, eps)

    elif mode == "test":
        x = (x - running_mean) / np.sqrt(running_var)
        out = x * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(d_out, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - d_out: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - d_gamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - d_beta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    N, D = d_out.shape
    out, x_norm, beta, gamma, xmu, ivar, std, var, eps = cache

    dx_norm = d_out * gamma
    d_ivar = np.sum(dx_norm * xmu, axis=0)
    dx_mu1 = dx_norm * ivar
    d_std = -1.0 / (std ** 2) * d_ivar
    d_var = 0.5 * 1. / np.sqrt(var + eps) * d_std
    dsq = 1.0 / N * np.ones((N, D)) * d_var
    dx_mu2 = 2 * xmu * dsq
    dx1 = dx_mu1 + dx_mu2
    d_mean = -1.0 * np.sum(dx1, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * d_mean
    dx = dx1 + dx2
    d_beta = np.sum(d_out, axis=0)
    d_gamma = np.sum(d_out * x_norm, axis=0)
    
    return dx, d_gamma, d_beta


def batchnorm_backward_alt(d_out, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalization backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    out, x_norm, beta, gamma, xmu, ivar, std, var, eps = cache
    N, D = d_out.shape

    d_gamma = np.diat(np.dot(d_out.T, x_norm))
    d_beta = np.sum(d_out, axis=0)

    d_out_dx = gamma * (1 - 1 / N) / ivar * (1 + 1 / N * ((out - beta) / gamma) ** 2)

    dx = d_out * d_out_dx

    return dx, d_gamma, d_beta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    ########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.    #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than six lines.   #
    ########################################################################

    x_swapped = np.transpose(x, (0, 2, 3, 1))
    x_swapped_reshaped = np.reshape(x_swapped, (-1, x_swapped.shape[-1]))

    out_temp, cache = batchnorm_forward(x_swapped_reshaped, gamma, beta, bn_param)
    out = np.transpose(np.reshape(out_temp, x_swapped.shape), (0, 3, 1, 2))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return out, cache


def spatial_batchnorm_backward(d_out, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - d_out: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - d_gamma: Gradient with respect to scale parameter, of shape (C,)
    - d_beta: Gradient with respect to shift parameter, of shape (C,)
    """
    ########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.   #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than six lines.   #
    ########################################################################

    d_out_swapped = np.transpose(d_out, (0, 2, 3, 1))
    d_out_swapped_reshaped = np.reshape(d_out_swapped, (-1, d_out_swapped.shape[-1]))

    dx_sr, d_gamma, d_beta = batchnorm_backward(d_out_swapped_reshaped, cache)
    dx = np.transpose(np.reshape(dx_sr, d_out_swapped.shape), (0, 3, 1, 2))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return dx, d_gamma, d_beta
