import os
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2 as cv
import threading

# Label colors, BGR format
CLASS_COLORS = np.array([
    (0, 0, 0),      # background (black)
    (0, 255, 0),    # tool wrist/gripper (green)
    (0, 0, 255),    # needle (red)
    (255, 0, 0),    # tool shaft (blue)
    (0, 255, 255),  # thread (yellow)
], dtype=np.uint8)

class SegmentationModelV2:
    def __init__(self):
        #package_path = rospkg.RosPack().get_path('accelnet_challenge_sdu')
        #path = os.path.join(package_path, 'resources', 'segmentation_model_v2')
        path = '../segmentation_model_v2'
        self.model = tf.keras.models.load_model(path, compile=False)
        self.lock = threading.Lock()

    def generate_masks(self, img):
        # For a single image, cv.copyMakeBorder is faster than np.pad.
        # For multiple stacked images (a batch), use np.pad instead.
        padded = cv.copyMakeBorder(img, top=4, bottom=4, left=0, right=0, borderType=cv.BORDER_CONSTANT)
        padded = tf.keras.applications.mobilenet_v2.preprocess_input(padded)

        # Make prediction
        # Re-entrancy OK, but allow only a single predict() call at a time to
        # avoid out-of-memory errors.
        with self.lock:
            pred = self.model.predict(padded[np.newaxis,...], verbose=0)  # np.newaxis because batch size = 1

        # Make BGR image with colors corresponding to predicted labels
        pred0 = pred[0][4:-4,...]  # first and only batch; cut away padding
        idx = np.argmax(pred0, axis=-1)
        labels = np.take(CLASS_COLORS, idx, axis=0)  # np.take() is faster than CLASS_COLORS[idx]

        return labels

def load_data():
    dataset_files = []
    for file in os.listdir("../Trainings_Data/vision_trainings_data"):
        dataset_files.append("../Trainings_Data/vision_trainings_data" + "/" + file)
        #print(file)
    print(f'Total files: {len(dataset_files)}')


    dataset_files = dataset_files[:50]
    #dataset_files = random.sample(dataset_files, 20)

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
        # image = image / 255.0  # Normalize the pixel values to the range 0-1.
        # image = image_preproccess(image)

        return image

    # Load the TFRecords file.
    full_dataset = tf.data.TFRecordDataset(dataset_files)

    # Limit the number of data pairs loaded to a specific value, e.g., 100.
    # limit = 2000
    # dataset = full_dataset.take(limit)
    dataset = full_dataset

    # Map the input data to the parser and decoder functions.
    dataset = dataset.map(_parse_image_function)
    # dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    return dataset


model_paths = [
    ("Fixed_RGB_eval.csv", "../../Models/FixedParametersCNNRGB", False, 270, 480, 3, True),
    ("Fixed_Segmentation_eval.csv", "../../Models/FixedParametersCNNSeg", True, 270, 480, 3, False),
    ("Variable_RGB_eval.csv", "../../Models/RGB_26", False, 270, 480, 3, False),
    ("Variable_Segmentation_eval.csv", "../../Models/Segmentation_107", True, 270, 480, 3, False),
    ("AlexNet_eval.csv", "RGB_Model_ALEXNET_V3", False, 270, 480, 3, False),
    ("AlexNet_RGB_random.csv", "RGB_Model_AlexNet_Random", False, 270, 480, 3, False),
    ("Fixed_RGB_random.csv", "RGB_Model_Fixed_Random", False, 270, 480, 3, False),
    ("AlexNet_segmented.csv", "Segmented_Model_AlexNet_10-06-2023_14-31-31", True, 270, 480, 3, False),
]

csv_file, model_name, run_segmentation, img_height, img_width, img_channel, normalize = model_paths[7]
dim = (img_width, img_height) #(480, 270)

dataset = load_data()
model = keras.models.load_model(f'models/{model_name}')



all_labels = []

# open the file in the write mode
f = open(csv_file, 'w')
# create the csv writer
writer = csv.writer(f)


if run_segmentation:
    # Initiated the segmentation once!
    print(f'Running with Segmentation')
    segmentation_model = SegmentationModelV2()

for i, (_image, label) in enumerate(dataset):
    # This is already normalized
    image = _image.numpy()
    if normalize:
        image = _image.numpy()/255.0
    true_label = label['output'].numpy()
    all_labels.append(true_label)

    if run_segmentation:
        seg_image = cv.resize(image, (1920, 1080))
        masked_image = segmentation_model.generate_masks(seg_image)
        # Resize the image for the model
        img_segmented = cv.resize(masked_image, dim)
        pred_label = model.predict(tf.expand_dims(img_segmented, axis=0), verbose=0)[0]
    else:
        pred_label = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]

    #print(f'target: {true_label} vs. {pred_label}')

    row = []
    for label_list in [true_label, pred_label]:
        for elem in label_list:
            row.append(float(elem))

    # write a row to the csv file
    writer.writerow(row)
    print(f'row: {row}')

f.close()


# open the file in the write mode
f = open("uniform_eval.csv", 'w')
# create the csv writer
writer = csv.writer(f)
for label in all_labels:
    # add noise to needle
    noisy_pos_needle = np.array(label, dtype=np.float32)
    #print(f'Before noise: {noisy_pos_needle}')
    mu, sigma = 0, 0.001  # mean and standard deviation
    noise = np.random.normal(mu, sigma, 7)
    noisy_pos_needle = noisy_pos_needle + noise
    #error = label - noisy_pos_needle

    row = []
    for label_list in [label, noisy_pos_needle]:
        for elem in label_list:
            row.append(float(elem))

    # write a row to the csv file
    writer.writerow(row)
    print(f'row: {row}')
f.close()






