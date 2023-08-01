import os
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt


img_height, img_width, img_channel = 270, 480, 3


dataset_files = []
for file in os.listdir("../Trainings_Data/vision_trainings_data"):
    dataset_files.append("../Trainings_Data/vision_trainings_data" + "/" + file)
    print(file)
print(f'Total files: {len(dataset_files)}')


#dataset_files = dataset_files[:50]
dataset_files = random.sample(dataset_files, 20)

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


#fig = plt.figure(figsize=(10, 10))
#ax = plt.axes(projection='3d')

xdata, ydata, zdata = [], [], []

for i, (image, label) in enumerate(dataset):
    pose = label['output'].numpy()
    xdata.append(pose[0])
    ydata.append(pose[1])
    zdata.append(pose[2])
    #if i % 24 == 0:
        #plt.scatter(xdata, ydata)
        #plt.plot(xdata, ydata)
        #ax.plot3D(xdata, ydata, 0)
        #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
        #xdata, ydata, zdata = [], [], []

# Assuming x and y are your data arrays
heatmap, xedges, yedges = np.histogram2d(xdata, ydata, bins=(15, 15))
cmap = plt.cm.get_cmap('hot')  # You can choose any colormap you prefer
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

fig = plt.figure(figsize=(10, 10))

plt.imshow(heatmap.T, cmap=cmap, extent=extent, origin='lower', interpolation='bilinear')
plt.colorbar()  # Add a colorbar for reference
#plt.xlabel('X-axis label')
#plt.ylabel('Y-axis label')
#plt.title('Heatmap')

fig.tight_layout()
fig.savefig(f'vision_figures/heatmap.jpg')
