"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections

import tensorflow as tf

flags = tf.app.flags

# Model dependent flags.

flags.DEFINE_integer('logits_kernel_size', 1,
                     'The kernel size for the convolutional kernel that '
                     'generates logits.')

# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65', we set atrous_rates = [6, 12, 18] (output stride 16)
# and decoder_output_stride = 4.
flags.DEFINE_enum('model_variant', 'xception_65',
                  ['xception_65'], 'DeepLab model variant.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,
                     'Use separable convolution for ASPP or not.')

flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
flags.DEFINE_integer('decoder_output_stride', None,
                     'The ratio of input to output spatial resolution when '
                     'employing decoder to refine segmentation results.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'

class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'atrous_rates',
        'output_stride',
        'merge_method',
        'add_image_level_feature',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              atrous_rates=None,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, atrous_rates, output_stride,
        FLAGS.merge_method, FLAGS.add_image_level_feature,
        FLAGS.aspp_with_batch_norm, FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid, FLAGS.decoder_output_stride,
        FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size,
        FLAGS.model_variant)
