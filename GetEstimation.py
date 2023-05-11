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
from tensorflow import keras

import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
from sensor_msgs.msg import Image

import segmentation_tensorflow as sdusegm

class estimation:
    def __init__(self, method):

        self.method = method
        self.model = None
        global_path_prefix = "accelnet-challenge-2022-main/src/accelnet_challenge_sdu"
        if self.method == "regression":
            self.model = tf.keras.models.load_model(f'{global_path_prefix}/models/regression_models/saved_model_sweep3_AlexNet_mine_2_2023-05-08 16:20:26.233458')
        elif self.method == "segmentation":
            self.model = tf.keras.models.load_model(f'{global_path_prefix}/models/segmented_regression_models/saved_model_sweep9_backbone_model_EfficientNet_7_2023-05-07 17:32:31.208974')

    def getEstimation(self, combinedImg):
        #saveImage = ImageSaver()
        #combinedImg = saveImage.save_image_data()

        #combinedImg = combinedImg/255

        #split image into left and right
        imgHeight = combinedImg.shape[0]
        imgWidth = combinedImg.shape[1]

        leftImg = combinedImg[0:int(imgHeight), 0:int(imgWidth/2)]
        rightImg = combinedImg[0:int(imgHeight), int(imgWidth/2):int(imgWidth)]

        dim = (480, 270)
        imgLeft = cv2.resize(leftImg, dim)
        imgRight = cv2.resize(rightImg, dim)

        #cv2.imwrite("test_before.jpg", imgLeft)
        imgLeft = (imgLeft/255.0)/255.0

        #imgLeft = tf.constant(imgLeft)
        #print(f'imgLeft: {imgLeft}')

        if self.method == "regression":
            #print(f'{tf.expand_dims(imgLeft, axis=0).shape} vs. {imgLeft.shape}')
            prediction = self.model.predict(tf.expand_dims(imgLeft, axis=0), verbose=0)[0]
            return prediction
        elif self.method == "segmentation":
            #Resize the image for the segmentation
            dimSeg = (1920, 1080)
            imgLeft = cv2.resize(leftImg, dimSeg)

            #do segmentation on the image
            segmentation_model = sdusegm.SegmentationModelV2()
            maskedLeftImage = segmentation_model.generate_masks(imgLeft)

            #Resize the image for the model
            imgSegmentedLeft = cv2.resize(maskedLeftImage, dim)

            prediction = self.model.predict(tf.expand_dims(imgSegmentedLeft, axis=0), verbose=0)[0]
            return prediction

