import cv2
from img_saver import ImageSaver
import os
import time
import random
import datetime
import numpy as np
import PyKDL
from PyKDL import Frame, Rotation, Vector
import tf2_ros
from enum import Enum
import rospy
import tensorflow as tf
#from tensorflow import keras

import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
from sensor_msgs.msg import Image
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dense
from keras.models import Model

import segmentation_tensorflow as sdusegm


# class Autoencoder(Model):
#     def __init__(self, latent_dim, img_height, img_width, img_channels):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#
#         self.encoder = tf.keras.Sequential([
#             # Flatten(),
#             # Dense(latent_dim, activation='relu'),
#             Input(shape=(img_height, img_width, img_channels)),
#             Conv2D(32, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2), padding='same'),
#             Conv2D(64, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2), padding='same'),
#             Conv2D(128, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2), padding='same'),
#             Flatten(),
#             Dense(latent_dim, activation='linear'),
#         ])
#
#         self.decoder = tf.keras.Sequential([
#             # Dense(784, activation='sigmoid'),
#             # Reshape((28, 28))
#             Dense(34 * 60 * 128),
#             Reshape((34, 60, 128)),
#             Conv2D(128, (3, 3), activation='relu', padding='same'),
#             UpSampling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu', padding='same'),
#             UpSampling2D((2, 2)),
#             Conv2D(32, (3, 3), activation='relu', padding='same'),
#             UpSampling2D((2, 2)),
#             Conv2D(img_channels, (3, 3), activation='sigmoid', padding='same')
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded




class estimation:
    def __init__(self, method):
        self.method = method
        self.model = None

        if self.method == "regression":
            print("Vision model: vision_pipeline/models/RGB_Model_ALEXNET_V3")
            self.model = tf.keras.models.load_model(f'vision_pipeline/models/RGB_Model_ALEXNET_V3')
        elif self.method == "segmentation":
            self.model = tf.keras.models.load_model(f'models/SegmentationTunedModel_V1')
            # Do segmentation on the image
            self.segmentation_model = sdusegm.SegmentationModelV2()
        elif self.method == "AE":
            #self.model = Autoencoder(latent_dim=12, img_height=272, img_width=480, img_channels=3)
            #self.model.build(input_shape=(None, 272, 480, 3))
            #self.model.built = True
            # autoencoder = tf.keras.models.load_model('models/AE_model.h5')
            #self.model.load_weights('vision_pipeline/AE_checkpoints/AE_model_ckp')
            self.model = tf.keras.models.load_model('vision_pipeline/AE_checkpoints/encoder_model.h5')
            # Do segmentation on the image
            self.segmentation_model = sdusegm.SegmentationModelV2()

    def getEstimation(self, input_image, method):

        imgHeight = input_image.shape[0]
        imgWidth = input_image.shape[1]
        print(f'image is: {imgHeight} X {imgWidth}')

        dim = (480, 270)

        if method == "segmentation":
            print(f'Running Vision with Segmentation model')
            masked_image = self.segmentation_model.generate_masks(input_image)

            # Resize the image for the model
            img_segmented = cv2.resize(masked_image, dim)
            #dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            #dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            #cv2.imwrite(f"Trainings_Data/vision_trainings_data/segmentation_test_{dt_string}.jpg", img_segmented)
            img_segmented = (img_segmented.astype('float32') / 255.0)

            prediction = self.model.predict(tf.expand_dims(img_segmented, axis=0), verbose=0)[0]
            return prediction

        if method == "regression":
            print(f'Running Vision with Regression model')
            # Resize an normalize image
            image = cv2.resize(input_image, dim)
            image = (image.astype('float32') / 255.0)

            prediction = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
            return prediction
        # elif self.method == "segmentation":

        if method == "AE":
            print(f'Running Vision with AutoEncoder')
            cv2.imwrite(f'img_dump/AE_img.jpg', input_image)

            masked_image = self.segmentation_model.generate_masks(input_image)
            cv2.imwrite(f'img_dump/AE_masked_image.jpg', masked_image)


            # Resize the image for the model
            img_segmented = cv2.resize(masked_image, (272, 480))
            img_segmented = (img_segmented.astype('float32') / 255.0)

            AE_embedding = self.model.predict(img_segmented.reshape(-1, 272, 480, 3))
            state_vector = AE_embedding[0]
            #state_vector = []
            #for elem in encoded_imgs[0]:
            #    state_vector.append(float(elem))
            print(f"Encoded image: {state_vector}")
            #state = self.model.predict(tf.expand_dims(img_segmented, axis=0), verbose=0)[0]

            #decoded_imgs = self.model.decoder(encoded_imgs)
            #reconstructed_image = decoded_imgs[0].numpy() * 255.0
            #reconstructed_image = reconstructed_image.reshape(272, 480, 3)

            #cv2.imwrite(f'img_dump/AE_decoded_imgs.jpg', reconstructed_image)

            return state_vector

