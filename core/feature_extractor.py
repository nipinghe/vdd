"""Extracts features for different models."""
import functools
import tensorflow as tf

import xception
slim = tf.contrib.slim


# A map from network name to network function.
networks_map = {
    'xception_65': xception.xception_65,
}

# A map from network name to network arg scope.
arg_scopes_map = {
    'xception_65': xception.xception_arg_scope,
}

# Names for end point features.
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features.
networks_to_feature_maps = {
    'xception_65': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    }
}

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = {
    'xception_65': 'xception_65',
}

def extract_features(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     num_classes=None,
                     global_pool=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    output_stride: The ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the network up to.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.

  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Unrecognized model variant.
  """
  if 'xception' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=1e-3,
        batch_norm_scale=True,
        regularize_depthwise=regularize_depthwise)
    features, end_points = get_network(
        model_variant, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=name_scope[model_variant])
  else:
    raise ValueError('Unknown model variant %s.' % model_variant)

  return features, end_points


def get_network(network_name, arg_scope=None):
  """Gets the network.

  Args:
    network_name: Network name.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.

  Returns:
    A network function that is used to extract features.

  Raises:
    ValueError: network is not supported.
  """
  if network_name not in networks_map:
    raise ValueError('Unsupported network %s.' % network_name)
  arg_scope = arg_scope or arg_scopes_map[network_name]()
  func = networks_map[network_name]
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(inputs, *args, **kwargs)
  return network_fn
