import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


img_height, img_width = 1080, 1920 #270, 480

# Define feature description dictionary
image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        #'pose': tf.io.FixedLenFeature([7], tf.float32),
        'pose': tf.io.VarLenFeature(tf.float32)
}

# Create a TFRecordDataset from the TFRecord file
dataset = tf.data.TFRecordDataset('../Trainings_Data/vision_trainings_data/vision_16-05-2023_17-15-50.tfrecords')
parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, image_feature_description))

# Iterate over the parsed dataset and print the contents
count = 1
for example in parsed_dataset:
    image_bytes = example['image'].numpy()
    pose_values = tf.sparse.to_dense(example['pose']).numpy()

    print(count)
    count+=1


    #print("Image bytes:", image_bytes)
    #print("Pose values:", pose_values)
    #print("---")

# print(dataset.shape)
# for record in dataset:
#     # Parse the example protocol buffer
#     example = tf.io.parse_single_example(record, feature_description)
#
#     # Get the image bytes and decode to an image
#     img_bytes = example['image'].numpy()
#
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#
#     # Get the filename
#     pose = example['pose'].numpy()
#
#     print(pose)

    # Display the image and filename
    #cv2.imshow("Image", img)
    #cv2.waitKey(500)
    #cv2.destroyAllWindows()