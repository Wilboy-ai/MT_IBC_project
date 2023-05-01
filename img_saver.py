"""
Script for projecting needle salient point in to the image plane

To use correctly this script make sure that:
   1) the /ambf/env/cameras/cameraL/ImageData topic is available
   2) the cameras can view the needle in the scene

Script tested in python 3.8 and ros Noetic

Juan Antonio Barragan 

"""

import json
import cv2
import numpy as np
from numpy.linalg import inv
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import time
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

np.set_printoptions(precision=3, suppress=True)


class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.left_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )
        self.right_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraR/ImageData", Image, self.right_callback
        )

        self.left_frame = None
        self.right_frame = None
        self.left_ts = None
        self.right_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def right_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_frame = cv2_img
            self.right_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    #def save_image(self, filename):
    #    if self.left_frame is not None and self.right_frame is not None:
    #        # Combine the left and right images horizontally
    #        combined_img = np.concatenate((self.left_frame, self.right_frame), axis=1)
    #        cv2.imwrite(filename, combined_img)
    #        print("Image saved to", filename)
    #    else:
    #        print("No image data available")

    def save_image(self, filename):
        while self.left_frame is None or self.right_frame is None:
            if self.left_frame is not None:
                print("Left image")
            if self.right_frame is not None:
                print("Right image")
            rospy.sleep(0.01)

        # Resize the left and right images to 960x540
        resized_left = cv2.resize(self.left_frame, (960, 540))
        resized_right = cv2.resize(self.right_frame, (960, 540))
        
        # Combine the left and right images horizontally
        combined_img = np.concatenate((resized_left, resized_right), axis=1)
        cv2.imwrite(filename, combined_img)
        print("Image saved to", filename)

    def save_image_data(self, type):
        while self.left_frame is None or self.right_frame is None:
            if self.left_frame is not None:
                print("Left image")
            if self.right_frame is not None:
                print("Right image")
            rospy.sleep(0.01)


        if type == 'stereo':
            # Resize the left and right images to 960x540
            resized_left = cv2.resize(self.left_frame, (960, 540))
            resized_right = cv2.resize(self.right_frame, (960, 540))

            # Combine the left and right images horizontally
            combined_img = np.concatenate((resized_left, resized_right), axis=1)
            resized_combined_img = cv2.resize(combined_img, (135, 480))
            gray_img = cv2.cvtColor(resized_combined_img, cv2.COLOR_BGR2GRAY)
            # Normalize the image
            normalized_img = gray_img.astype('float32') / 255.0
        else:
            resized_right = cv2.resize(self.right_frame, (120, 68))
            gray_img = cv2.cvtColor(resized_right, cv2.COLOR_BGR2GRAY)
            # Normalize the image
            #normalized_img = gray_img.astype('float32') / 255.0

        # Flatten the image
        #flattened_img = normalized_img.flatten()
        return resized_right

    def get_frame(self):
        while self.right_frame is None:
            if self.right_frame is not None:
                print("Right image is None")
            rospy.sleep(0.01)
        # Resize the left and right images to 960x540
        resized_right = cv2.resize(self.right_frame, (480, 270))
        return resized_right



#print("Project Needle points!")


# if __name__ == "__main__":
#     saver = ImageSaver()
#     for i in range(0, 3):
#         #saver.save_image("image.jpg")
#
#         #Get image
#         img = saver.left_frame
#
#         #Check if image is available
#         if saver.left_frame is not None:
#            cv2.imshow("img", img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#         else:
#            print("Image is not available yet")
#
#         rospy.sleep(1.0)

