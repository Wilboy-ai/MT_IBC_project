# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)

# If the list of devices is not specified in
# `tf.distribute.MirroredStrategy` constructor, they will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))







