import tensorflow as tf
from tensorflow import keras
import wandb
import random
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation
import cv2
import os
import time
import datetime
#from datetime import datetime
from scipy.stats import norm

import cv2

import numpy as np
import PyKDL
from PyKDL import Frame, Rotation, Vector
import tf2_ros
from enum import Enum
import rospy

import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
from sensor_msgs.msg import Image

import segmentation_tensorflow as sdusegm

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
   try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)]) #only go upto 8000
   except RuntimeError as e:
      print(e)

def load_dataset(img_height, img_width, img_channel):
    dataset_files = []
    for file in os.listdir("Trainings_Data/vision_trainings_data"):
        dataset_files.append("Trainings_Data/vision_trainings_data" + "/" + file)
        print(file)
    print(f'Total files: {len(dataset_files)}')

    #dataset_files = random.sample(dataset_files, 100)
    dataset_files = dataset_files[:50]

    # Define the input data format.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'pose': tf.io.FixedLenFeature([7], tf.float32),
    }

    # Define a function to parse the input data.
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    # Define a function to decode the image and normalize the pixel values.
    def decode_image(image):
        image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_height, img_width])  # Resize the image to a fixed size.
        #image = image / 255.0  # Normalize the pixel values to the range 0-1.

        return image

    # Load the TFRecords file.
    full_dataset = tf.data.TFRecordDataset(dataset_files)

    # Limit the number of data pairs loaded to a specific value, e.g., 100.
    #limit = 2000
    #dataset = full_dataset.take(limit)
    dataset = full_dataset

    # Map the input data to the parser and decoder functions.
    dataset = dataset.map(_parse_image_function)
    #dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    return dataset

#Segment the image
def segment(image, progress):
    dim = (img_width, img_height)

    image = np.array(image)
    image = image * 255.0
    #cv2.imwrite(f"Plots/before{progress}.jpg", image)

    segmentation_model = sdusegm.SegmentationModelV2()
    masked_image = segmentation_model.generate_masks(image)
    #cv2.imwrite(f"Plots/after{progress}.jpg", masked_image)

    masked_image = cv2.resize(masked_image, dim)
    #masked_image = np.array(masked_image)
    masked_image = masked_image / 255.0
    #print(masked_image)

    #For the FPCNN double normalisation is needed
    masked_image = masked_image / 255.0

    return masked_image

def evaluate_model(img_height, img_width, img_channel, model_name):
    dataset = load_dataset(img_height, img_width, img_channel=img_channel)
    #model = keras.models.load_model(f'Models/{model_name}')
    model = keras.models.load_model(model_name)

    x_err = []
    y_err = []
    z_err = []
    x_orr_err = []
    y_orr_err = []
    z_orr_err = []
    w_err = []

    progress = 1

    for i, (image, label) in enumerate(dataset):
        true_label = label['output'].numpy()

        # I would like to make a function call to run segmentation on all the images here
        #image = segment(image, progress)

        # normCheck = pow(true_label[3], 2) + pow(true_label[4], 2) + pow(true_label[5], 2) + pow(true_label[6], 2)
        # print(f"Normalisation check: {normCheck}")

        pred_label = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]

        # print(f"True label: {true_label}")
        # print(f"Predicted label: {pred_label}")

        true_values = np.array(true_label)
        predicted_values = np.array(pred_label)

        error_array = np.subtract(true_values, predicted_values)
        error = list(error_array)

        x_err.append(error[0])
        y_err.append(error[1])
        z_err.append(error[2])
        x_orr_err.append(error[3])
        y_orr_err.append(error[4])
        z_orr_err.append(error[5])
        w_err.append(error[6])

        print(progress)
        progress += 1

    totalTranErr = x_err + y_err + z_err
    meanTranErr = np.mean(np.absolute(totalTranErr))
    print(f"The MAE in translation: {meanTranErr}")

    totalOriErr = x_orr_err + y_orr_err + z_orr_err + w_err
    meanOriErr = np.mean(np.absolute(totalOriErr))
    print(f"The MAE in orientation: {meanOriErr}")

    metricNames = ["x", "y", "z", "x_orientation", "y_orientation", "z_orientation", "w_orientation"]
    for name in metricNames:
        if name == "x":
            data = x_err
            unit = "meters"
        if name == "y":
            data = y_err
            unit = "meters"
        if name == "z":
            data = z_err
            unit = "meters"
        if name == "x_orientation":
            data = x_orr_err
            unit = "radians"
        if name == "y_orientation":
            data = y_orr_err
            unit = "radians"
        if name == "z_orientation":
            data = z_orr_err
            unit = "radians"
        if name == "w_orientation":
            data = w_err
            unit = "radians"

        data_arr = np.array(data)

        fig, ax = plt.subplots()

        binNum = 25

        n, bins, patches = ax.hist(data, binNum, density=1)

        # mu = data_arr.mean()
        # sigma = data_arr.std()

        mu, sigma = norm.fit(data)

        # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        #     np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))

        # y = (1 / (np.sqrt(2) * np.pi * sigma)) * np.exp(-1 * (pow(bins - mu, 2) / (2 * sigma)))

        #ax.plot(bins, norm.pdf(bins, mu, sigma), color='black')

        textstr = '\n'.join((
            r'$\mu=%.4f$' % (mu,),
            r'$\sigma=%.4f$' % (sigma,)))

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        plt.xlabel(f'Error in {unit}')
        plt.ylabel('Number of occurences')

        plt.title('Error in the ' + name + ' direction',
                  fontweight="bold")
        plt.savefig('Plots/AlexNet/' + name + 'Error.png')


img_height, img_width, img_channel = 270, 480, 1 #270, 480 #540, 960  # 1080, 1920

evaluate_model(img_height, img_width, img_channel, "vision_pipeline/models/RGB_Model_ALEXNET_V3")






