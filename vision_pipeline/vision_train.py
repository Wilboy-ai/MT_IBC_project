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
#import cv2
import os
import time
import datetime
#from datetime import datetime
from scipy.stats import norm
import create_model
import tensorflow as tf
import cv2 as cv
import threading


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
   try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)]) #only go upto 8000
   except RuntimeError as e:
      print(e)

SEGMENT = True

# # Label colors, BGR format
# CLASS_COLORS = np.array([
#     (0, 0, 0),      # background (black)
#     (0, 255, 0),    # tool wrist/gripper (green)
#     (0, 0, 255),    # needle (red)
#     (255, 0, 0),    # tool shaft (blue)
#     (0, 255, 255),  # thread (yellow)
# ], dtype=np.uint8)

# Label colors, BGR format
CLASS_COLORS = np.array([
    (0, 0, 0),      # background (black)
    (0, 0, 0),    # tool wrist/gripper (green)
    (255, 255, 255),    # needle (red)
    (0, 0, 0),    # tool shaft (blue)
    (0, 0, 0),  # thread (yellow)
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

if SEGMENT:
    # Initiated the segmentation once!
    print(f'Running with Segmentation')
    segmentation_model = SegmentationModelV2()



# def image_preproccess(_image):
#     image = tf.keras.backend.eval(_image)*255
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Define the red threshold (adjust as needed)
#     red_threshold = 90
#
#     # Remove pixels with high red values
#     # Split the image into channels
#     r, g, b = cv2.split(image)
#
#     # Apply thresholding to the red channel
#     _, red_mask = cv2.threshold(r, red_threshold, 255, cv2.THRESH_BINARY_INV)
#
#     # Convert the red mask to the same data type as the image
#     red_mask = red_mask.astype(np.uint8)
#
#     # Resize the red mask to match the dimensions of the image
#     red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))
#
#     return red_mask


def load_dataset(img_height, img_width, img_channel):
    dataset_files = []
    for file in os.listdir("../Trainings_Data/vision_trainings_data"):
        dataset_files.append("../Trainings_Data/vision_trainings_data" + "/" + file)
        print(file)
    print(f'Total files: {len(dataset_files)}')

    #dataset_files = dataset_files[:25]
    dataset_files = random.sample(dataset_files, 50)

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
        print(image)
        print(type(image))

        image = tf.image.decode_jpeg(image, channels=3)
        if img_channel == 1:
            # greyscale image
            print("greyscaling image")
            image = tf.image.rgb_to_grayscale(image)  # Convert the image to grayscale.

        image = tf.image.convert_image_dtype(image, tf.float32)
        if SEGMENT:
            image = tf.image.resize(image, [1080, 1920])  # Resize the image to a fixed size.
        else:
            image = tf.image.resize(image, [img_height, img_width])  # Resize the image to a fixed size.

        # image = image / 255.0  # Normalize the pixel values to the range 0-1.
        # image = image_preproccess(image)

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

    segmented_images = []
    needle_poses = []

    if SEGMENT:
        print("Segmenting image")

        for i, (image, label) in enumerate(dataset):
            print(f'Processing image {i}')
            img = image.numpy()
            img *= 255.0
            #img = cv.resize(img, (1920, 1080))
            segmented_image = segmentation_model.generate_masks(img)
            segmented_image = cv.resize(segmented_image, (img_width, img_height))
            #cv.imwrite(f'vision_figures/segmented_image_{i}.jpg', segmented_image)
            segmented_images.append(segmented_image/255.0)
            needle_poses.append(label['output'].numpy())

        # Apply image segmentation to each image in the dataset
        #dataset = dataset.map(lambda image, labels: (segment_image(image), labels))

        # Convert the arrays to TensorFlow tensors
        print("Converting segmented_images_tensor")
        segmented_images_tensor = tf.convert_to_tensor(segmented_images)
        print("Converting needle_poses_tensor")
        needle_poses_tensor = tf.convert_to_tensor(needle_poses)

        # Create a dataset from the tensors
        print("Converting from_tensor_slices")
        dataset = tf.data.Dataset.from_tensor_slices((segmented_images_tensor, needle_poses_tensor))
        return dataset

    #dataset = dataset.map(segmented_images, needle_poses)
    return dataset


def train(img_height, img_width, img_channel):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Vision_Pipeline",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1.e-6,
            "batch": 8,
            "epochs": 25,
            "val_split": 0.3,
        }
    )

    # datasetFiles = []
    # count = 0
    # for file in os.listdir("../Trainings_Data/vision_trainings_data"):
    #     datasetFiles.append("../Trainings_Data/vision_trainings_data" + "/" + file)
    #     print(file)
    #     #count += 1
    #     #if count >= 200:
    #     #    break
    #
    # print(f'Total files: {len(datasetFiles)}')
    # dataset_files = random.sample(datasetFiles, 200)
    # print(f'Sampled files: {len(dataset_files)}')

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        print("Device:", tpu.master())
        strategy = tf.distribute.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print("Number of replicas:", strategy.num_replicas_in_sync)

    learning_rate = wandb.config.learning_rate
    batch = wandb.config.batch
    epochs = wandb.config.epochs
    val_split = wandb.config.val_split

    # img_height, img_width = 270, 480 #1080, 1920
    # scale = 1
    #
    # # Define the input data format.
    # image_feature_description = {
    #     'image': tf.io.FixedLenFeature([], tf.string),
    #     'pose': tf.io.FixedLenFeature([7], tf.float32),
    # }
    #
    # # Define a function to parse the input data.
    # def _parse_image_function(example_proto):
    #     return tf.io.parse_single_example(example_proto, image_feature_description)
    #
    # # Define a function to decode the image and normalize the pixel values.
    # def decode_image(image):
    #     image = tf.image.decode_jpeg(image, channels=3)
    #     image = tf.image.convert_image_dtype(image, tf.float32)
    #     image = tf.image.resize(image, [img_height, img_width])  # Resize the image to a fixed size.
    #     #image = image / 255.0  # Normalize the pixel values to the range 0-1.
    #     return image
    #
    # # Load the TFRecords file.
    # dataset = tf.data.TFRecordDataset(datasetFiles)
    #
    # # Map the input data to the parser and decoder functions.
    # dataset = dataset.map(_parse_image_function)
    # #dataset = dataset.map(lambda x: {'input_1': decode_image(x['image']), 'output': x['pose']})
    # #dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose'][0:3]*scale}))
    # dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    #
    # for i, (image, x) in enumerate(dataset):
    #     print(f'{i}: {x}')
    #     print(x)
    #     break
    #
    # # Calculate the number of samples in the dataset.
    # num_samples = dataset.reduce(0, lambda x, _: x + 1).numpy()
    # print(f'Total: {num_samples}')
    #
    # # Shuffle and split the dataset into training and validation sets.
    # dataset = dataset.shuffle(num_samples)

    # Set dims
    #img_height, img_width = 270, 480  # 1080, 1920
    dataset = load_dataset(img_height, img_width, img_channel)
    print("Got here!")
    num_samples = dataset.reduce(0, lambda x, _: x + 1).numpy()
    print(f'Total: {num_samples}')
    dataset = dataset.shuffle(num_samples)

    # Split the dataset into training and validation sets.
    val_size = int(val_split * num_samples)
    train_dataset = dataset.skip(val_size)
    val_dataset = dataset.take(val_size)

    print(f'train: {train_dataset.reduce(0, lambda x, _: x + 1).numpy()}')
    print(f'valid: {val_dataset.reduce(0, lambda x, _: x + 1).numpy()}')

    # Batch the datasets for training and validation.
    train_dataset = train_dataset.batch(batch)
    val_dataset = val_dataset.batch(batch)

    #def checkImgNorm(image):
        #maxValue = np.max(image * 255)
        #if maxValue == 255:
        #    print("This image looks okay")
        #    return True
        #print("This image does not look okay, trying another image")
        #return False

    #testImage = None
    for i, (image, label) in enumerate(dataset):
        print(image)

        maxValue = np.max(image * 255)
        if maxValue == 255:
            print("This image looks okay")
            break
        print("This image does not look okay, trying another image")
        if i >= 10:
            raise Exception("Wrong format of images in dataset")


    #assert checkImgNorm(testImage)

    #model = create_model.VGG16(img_height=img_height, img_width=img_width, img_channel=img_channel)
    model = create_model.create_AlexNet(img_height=img_height, img_width=img_width, img_channel=img_channel)
    #model = create_model.create_MT_Model(img_height, img_width)
    #model = create_model.create_MT_Model_V2(img_height, img_width)
    #model = create_model.create_backbone_model_ResNet(img_height, img_width)
    #model = create_model.create_backbone_model_EfficientNet(img_height, img_width)
    #model = create_model.CNN(img_height, img_width, depth, kernels, kernelSize, dropout, phases, pooling, kernelStep, fcDepth, fcWidth)
    #model = create_model.create_RESNET(img_height, img_width, num_residual_blocks=10)

    #model = create_model.create_split_model(img_height, img_width)

    # Load model weights
    #model.load_weights("RGB_Model_EffiecientNet_08-06-2023_11-47-56.h5")


    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #     loss={'output': keras.losses.MeanSquaredError()},
    #     #metrics=["accuracy"],
    #     metrics=[keras.metrics.MeanAbsoluteError(),],
    # )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'output': keras.losses.MeanAbsoluteError()},
        # metrics=["accuracy"],
        metrics=[keras.metrics.MeanAbsoluteError(), ],
    )

    model.summary()

    # for epoch in range(epochs):
    #      history = model.fit(train_dataset, batch_size=1, epochs=1, verbose=1, validation_data=val_dataset, shuffle=True)
    #      if epoch == 0:
    #          print(history.history)
    #
    #      MAE = history.history['mean_absolute_error'][0]
    #      val_MAE = history.history['val_mean_absolute_error'][0]
    #      MAE_loss = history.history['loss'][0]
    #      val_MAE_loss = history.history['val_loss'][0]
    #
    #      wandb.log({'MAE': MAE, 'MAE_loss': MAE_loss, 'val_MAE_loss': val_MAE_loss, 'val_MAE': val_MAE})

    history = model.fit(train_dataset, batch_size=batch, epochs=epochs, verbose=1, validation_data=val_dataset, shuffle=True)
    #print(history.history)

    MAE = history.history['mean_absolute_error']
    val_MAE = history.history['val_mean_absolute_error']
    MAE_loss = history.history['loss']
    val_MAE_loss = history.history['val_loss']

    for i in range(len(MAE)):
        wandb.log({'MAE': MAE[i], 'MAE_loss': MAE_loss[i], 'val_MAE_loss': val_MAE_loss[i], 'val_MAE': val_MAE[i]})

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    modelName=f'models/Segmented_Model_AlexNet_{dt_string}'
    #modelName = f'models/RGB_Model_EffiecientNet'
    print(f'saving model: {modelName}')
    model.save(modelName)

    # Save model weights
    #model.save_weights(f"{modelName}.h5")



def evaluate_model(img_height, img_width, img_channel, model_name):
    #model_name = "RGB_Model_V2"
    dataset = load_dataset(img_height, img_width, img_channel=img_channel)
    model = keras.models.load_model(f'models/{model_name}')
    def plot_example(plot_name, image, true_label, pred_label):
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title('True Pose: {}\nPredicted Pose: {}'.format(true_label, pred_label))
        plt.axis('off')

        # Save the plot as a JPEG.
        plt.savefig(f'vision_figures/image_{plot_name}.jpg')
        plt.clf()
        # plt.show()

    all_errors = []
    for i, (image, label) in enumerate(dataset):
        #print(f'Rendered image {i}')
        true_label = label['output'].numpy()
        pred_label = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]

        #error = np.abs(true_label - pred_label)
        error = true_label - pred_label
        all_errors.append(error)

        # Save the plot as a JPEG.
        #plot_example(i + 1, image.numpy(), true_label, pred_label)

        #if i >= 50:
        #    break

    dimensions = len(label['output'].numpy())

    fig, axs = plt.subplots(1, dimensions, figsize=(40, 5))
    #fig.suptitle('Vertically stacked subplots')
    titles = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    for i in range(0, dimensions):
        all_errors = np.array(all_errors)
        axs[i].hist(all_errors[:, i],  density=True, bins=10, alpha=0.7)

        # Estimate the parameters of the normal distribution
        mean, std = norm.fit(all_errors[:, i])
        # Plot the estimated normal distribution
        x = np.linspace(all_errors[:, i].min(), all_errors[:, i].max(), 100)
        y = norm.pdf(x, mean, std)
        axs[i].plot(x, y, 'r-', linewidth=2)

        # Add mean and standard deviation to the subplot
        axs[i].text(0.02, 0.98, f"Mean: {mean:.5f}\nStd: {std:.5f}", transform=axs[i].transAxes,
                    ha='left', va='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
        # Add subtitle to the subplot
        axs[i].set_title(f'{titles[i]}')

    fig.tight_layout()
    fig.savefig(f'vision_figures/{model_name}_Histogram.jpg')


# def image_prepross_test():
#     # Set dims
#     img_height, img_width = 270, 480  # 1080, 1920
#     dataset = load_dataset(img_height, img_width)
#     for i, (_image, label) in enumerate(dataset):
#         image = _image.numpy() * 255
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(f"vision_figures/Segmented_image.jpg", image)
        # # Convert image to grayscale
        # #gray_image = cv2.cvtColor(image.numpy()*255, cv2.COLOR_RGB2GRAY)
        # #hsv_image = cv2.cvtColor(image.numpy()*255, cv2.COLOR_RGB2HSV)
        # #hsl_image = cv2.cvtColor(image.numpy()*255, cv2.COLOR_RGB2HLS)
        #
        # # # Convert grayscale image to 8-bit unsigned integer
        # # gray_image = np.uint8(hsv_image)
        # #
        # # # Manually define the threshold value (e.g., 127)
        # # threshold_value = 127
        # #
        # # # Thresholding to keep only gray pixels using the manual threshold value
        # # _, segmented_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        # #
        # # # Thresholding to keep only gray pixels
        # # _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #
        # # Save the segmented image
        #
        #
        # # Define the red threshold (adjust as needed)
        # red_threshold = 90
        #
        # # Remove pixels with high red values
        # # Split the image into channels
        # r, g, b = cv2.split(image)
        #
        # # Apply thresholding to the red channel
        # _, red_mask = cv2.threshold(r, red_threshold, 255, cv2.THRESH_BINARY_INV)
        #
        # # Convert the red mask to the same data type as the image
        # red_mask = red_mask.astype(np.uint8)
        #
        # # Resize the red mask to match the dimensions of the image
        # red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))
        #
        # # Merge the modified red channel with the original green and blue channels
        # #modified_image = cv2.merge((b, g, cv2.bitwise_and(r, red_mask)))
        #
        # # Apply the red mask to remove red pixels from the original image
        # modified_image = cv2.bitwise_and(image, image, mask=red_mask)
        #
        # cv2.imwrite("vision_figures/preprocessed_image.jpg", modified_image)
        # cv2.imwrite("vision_figures/preprocessed_image_mask.jpg", red_mask)
        # #if i >= 30:
        # #    break


img_height, img_width = 270, 480 #270, 480 #540, 960  # 1080, 1920
if SEGMENT:
    img_channel = 1
else:
    img_channel = 3

#image_prepross_test()
train(img_height, img_width, img_channel)
#evaluate_model(img_height, img_width, img_channel, "RGB_Model_ALEXNET_linear_18-05-2023_10-01-20")



import csv
def plot_needle_csv():
    path = "../Trainings_Data/needle_pose_test.csv"
    all_data = []
    with open(path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        for row in plots:
            #print(np.array(row))
            all_data.append(row)

            #for elem in row:
            #    print(elem)
            #x.append(row[0])
            #y.append(int(row[2]))
    all_data = np.array(all_data)

    for i in range(0, 7):
        print(all_data[0:6,i])
        #plt.plot(all_data[0:6, i], label=f"dim:{i}")
        #plt.hist(all_data[0:6, i], density=True, bins=10, alpha=0.7)
        #plt.title('Ages of different persons')
    #plt.legend()
    #plt.show()


#plot_needle_csv()







# sweep_configuration = {
#     'name': 'CNN_model',
#     'method': 'bayes',
#     'metric': {
#         'name': 'val_MAE_loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         #'batch': {'values': [1, 8, 16, 32]},
#         'batch': {'values': [1, 8, 16]},
#         #'learning_rate': {'max': 1.e-5, 'min': 1.e-7},
#         #'learning_rate': {'values': [1.e-6]},
# 	'learning_rate': {'values': [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7]},
#         #'depth': {'max': 5, 'min': 1},
#         'depth': {'values': [3]},
#         #'kernels': {'max': 128, 'min': 1},
#         'kernels': {'values': [1, 8, 16, 32]},
#         #'kernelSize': {'max': 5, 'min': 1},
#         'kernelSize': {'values': [3]},
#         #'dropout': {'max': 0.9, 'min': 0.0},
#         'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
#         #'phases': {'max': 3, 'min': 1},
#         'phases': {'values': [2]},
#         #"'pooling': {'max': 5, 'min': 0},
#         'pooling': {'values': [3]},
#         #'kernelStep': {'max': 5, 'min': 1},
#         'kernelStep': {'values': [3, 4, 5]},
#         #'fcDepth': {'max': 10, 'min': 1},
#         'fcDepth': {'values': [5, 6, 7, 8, 9, 10]},
#         #'fcWidth': {'max': 100, 'min': 1},
#         'fcWidth': {'values': [50, 60, 70, 80, 90, 100]},
#     }
# }

# sweep_id = wandb.sweep(
#     sweep = sweep_configuration,
#     project = "CNN_model"
# )
#
# wandb.agent(sweep_id, function=main)

# [optional] finish the wandb run, necessary in notebooks
#wandb.finish()

# Define a function to plot an example from the dataset.


#
#
# def plot_example(plot_num, image, true_label, pred_label):
#     # Extract RPY angles and XYZ vectors from input poses
#     x1, y1, z1, R1, P1, Y1 = true_label
#     x2, y2, z2, R2, P2, Y2 = pred_label
#
#     # Convert RPY angles to rotation matrices
#     r1 = Rotation.from_euler('xyz', [R1, P1, Y1], degrees=False).as_matrix()
#     r2 = Rotation.from_euler('xyz', [R2, P2, Y2], degrees=False).as_matrix()
#
#     # Compute relative rotation matrix
#     rel_rot = r2.dot(r1.T)
#     # Compute angular error
#     #rot_error = Rotation.from_matrix(rel_rot).magnitude()
#     r, _ = cv2.Rodrigues(rel_rot)
#     rot_error = np.linalg.norm(r)
#
#     # Compute distance error
#     trans_error = np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])
#
#     # Create the plot
#     fig, axs = plt.subplots(1, 1, figsize=(10, 5))
#
#     # Set the plot title
#     axs.set_title('Pose Error: {}'.format(plot_num))
#
#     # Display the image
#     axs.imshow(image * 255)
#     axs.axis('off')
#
#     # Add text for the rotation and translation error
#     axs.text(0.05, 0.05, 'Rotation Error: {:.3f} [radians]\nTranslation Error: {:.3f}[meters]'.format(rot_error, trans_error),
#              bbox=dict(facecolor='white', alpha=0.5),
#              horizontalalignment='center',
#              transform=axs.transAxes)
#
#     # Adjust the position of the axes to make room for the text box
#     fig.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.9)
#
#     # Save the plot as a JPEG.
#     if plot_num <= 10:
#         plt.savefig(f'{save_data_path}/{plot_name_prefix}_{plot_num}.jpg', bbox_inches='tight')
#         plt.clf()
#         print(f'Plot {plot_num} being created')
#     return rot_error, trans_error

# Evaluate the trained model on all examples from the dataset.
# rot_errors = []
# trans_errors = []
# for i, (image, label) in enumerate(dataset):
#     true_label = label['output'].numpy()
#     pred_label = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
#
#     # Save the plot as a JPEG.
#     rot_error, trans_error = plot_example(i + 1, image.numpy(), true_label, pred_label)
#
#     rot_errors.append(rot_error)
#     trans_errors.append(trans_error)

# Create box plots
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#
# # Rotation error box plot
# axs[0].boxplot(rot_errors)
# axs[0].set_title('Rotation Error')
# axs[0].set_ylabel('Absolute Error (radians)')
#
# # Translation error box plot
# axs[1].boxplot(trans_errors)
# axs[1].set_title('Translation Error')
# axs[1].set_ylabel('Absolute Error (meters)')

# Save the plot as a JPEG.
# plt.savefig(f'{save_data_path}/{plot_name_prefix}_errors_boxplots.jpg')
# plt.clf()










# Evaluate the trained model on a couple of examples from the dataset.
#index = 1
#for image, label in dataset: #dataset.take(10):
#    true_label = label['output'].numpy()
#    pred_label = model.predict(tf.expand_dims(image, axis=0))[0]
#    plot_example(index, image.numpy(), true_label, pred_label)
#    index = index + 1

# Define a function to plot an example from the dataset.
#def plot_example(example):
#    image, label = example
#    plt.imshow(image.numpy())
#    plt.title('Pose: {}'.format(label['output'].numpy()))
#    plt.axis('off')
#    plt.show()

# Plot a couple of examples from the dataset.
#for example in dataset.take(2):
#    plot_example(example)





