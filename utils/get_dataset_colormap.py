"""Visualizes the segmentation results via specified color map.

Visualizes the semantic segmentation results by the color map
defined by the different datasets. Supported colormaps are:
"""

import numpy as np

# Dataset names.
_CAR_DAMAGE = 'car_damage'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _CAR_DAMAGE: 256
}


def bit_get(val, idx):
  """Gets the bit value.

  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_256_label_colormap():
  """Creates a label colormap used in general dataset.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((_DATASET_MAX_ENTRIES[_CAR_DAMAGE], 3), dtype=int)
  ind = np.arange(_DATASET_MAX_ENTRIES[_CAR_DAMAGE], dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
    raise ValueError('label value too large.')

  colormap = create_256_label_colormap()
  return colormap[label]
