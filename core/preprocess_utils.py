"""Utility functions related to preprocessing inputs."""
import tensorflow as tf


def flip_dim(tensor_list, prob=0.5, dim=1):
  """Randomly flips a dimension of the given tensor.

  The decision to randomly flip the `Tensors` is made together. In other words,
  all or none of the images pass in are flipped.

  Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
  that we can control for the probability as well as ensure the same decision
  is applied across the images.

  Args:
    tensor_list: A list of `Tensors` with the same number of dimensions.
    prob: The probability of a left-right flip.
    dim: The dimension to flip, 0, 1, ..

  Returns:
    outputs: A list of the possibly flipped `Tensors` as well as an indicator
    `Tensor` at the end whose value is `True` if the inputs were flipped and
    `False` otherwise.

  Raises:
    ValueError: If dim is negative or greater than the dimension of a `Tensor`.
  """
  random_value = tf.random_uniform([])

  def flip():
    flipped = []
    for tensor in tensor_list:
      if dim < 0 or dim >= len(tensor.get_shape().as_list()):
        raise ValueError('dim must represent a valid dimension.')
      flipped.append(tf.reverse_v2(tensor, [dim]))
    return flipped

  is_flipped = tf.less_equal(random_value, prob)
  outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
  if not isinstance(outputs, (list, tuple)):
    outputs = [outputs]
  outputs.append(is_flipped)

  return outputs



