# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense Resnet Value Network."""
import tensorflow.compat.v2 as tf
import tensorflow as tf

class DenseResnetValue(tf.keras.layers.Layer):
  """Dense Resnet layer."""

  def __init__(self, width=512, num_blocks=2):
    super(DenseResnetValue, self).__init__()
    self.dense0 = dense(width)
    self.blocks = [ResNetDenseBlock(width) for _ in range(num_blocks)]
    self.dense1 = dense(1)

  def call(self, x, training):
    x = self.dense0(x, training=training)
    for block in self.blocks:
      x = block(x, training=training)
    x = self.dense1(x, training=training)
    return x


def dense(width):
  """Linear layer, no activation."""
  return tf.keras.layers.Dense(
      width,
      activation=None,
      kernel_initializer='normal',
      bias_initializer='normal')


class ResNetDenseBlock(tf.keras.layers.Layer):
  """Dense resnet block."""

  def __init__(self, width):
    super(ResNetDenseBlock, self).__init__()
    self.dense0 = dense(width // 4)
    self.dense1 = dense(width // 4)
    self.dense2 = dense(width)
    self.dense3 = dense(width)

    self.activation0 = tf.keras.layers.ReLU()
    self.activation1 = tf.keras.layers.ReLU()
    self.activation2 = tf.keras.layers.ReLU()
    self.activation3 = tf.keras.layers.ReLU()

  def call(self, x, training):
    y = self.dense0(self.activation0(x))
    y = self.dense1(self.activation1(y))
    y = self.dense2(self.activation2(y))
    if x.shape != y.shape:
      x = self.dense3(self.activation3(x))
    return x + y


"""Simple 4-layer conv + maxpool CNN."""
def get_conv_maxpool(target_height, target_width, nchannels):
  """Instantiates simple cnn architecture."""
  # Determine proper input shape
  input_shape = (target_height, target_width, nchannels)
  img_input = tf.keras.layers.Input(shape=input_shape)
  # [96, 128, 32].
  x = tf.keras.layers.Conv2D(
      32, 3, padding='same', activation=tf.nn.relu, name='conv1')(img_input)
  # [96, 128, 32].
  x = tf.keras.layers.MaxPool2D()(x)
  # [48, 64, 32].
  x = tf.keras.layers.Conv2D(
      64, 3, padding='same', activation=tf.nn.relu, name='conv2')(x)
  # [48, 64, 64].
  x = tf.keras.layers.MaxPool2D()(x)
  # [24, 32, 64].
  x = tf.keras.layers.Conv2D(
      128, 3, padding='same', activation=tf.nn.relu, name='conv3')(x)
  # [24, 32, 128].
  x = tf.keras.layers.MaxPool2D()(x)
  # [12, 16, 128].
  x = tf.keras.layers.Conv2D(
      256, 3, padding='same', activation=tf.nn.relu, name='conv4')(x)
  # [12, 16, 256].
  x = tf.keras.layers.MaxPool2D()(x)
  # [6, 8, 256].
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  # [256]
  return tf.keras.models.Model(img_input, x, name='conv_maxpool')