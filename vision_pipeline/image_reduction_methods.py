import numpy as np
from sklearn.decomposition import PCA
import os
import tensorflow as tf
import random
#import cv2
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import cv2 as cv
import threading
import pickle

# Label colors, BGR format
CLASS_COLORS = np.array([
    (0, 0, 0),      # background (black)
    (0, 255, 0),    # tool wrist/gripper (green)
    (0, 0, 255),    # needle (red)
    (255, 0, 0),    # tool shaft (blue)
    (0, 255, 255),  # thread (yellow)
], dtype=np.uint8)

# # Label colors, BGR format
# CLASS_COLORS = np.array([
#     (0, 0, 0),      # background (black)
#     (0, 0, 0),    # tool wrist/gripper (green)
#     (0, 0, 255),    # needle (red)
#     (0, 0, 0),    # tool shaft (blue)
#     (0, 0, 0),  # thread (yellow)
# ], dtype=np.uint8)



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


# Define the dimensions of the input data
img_height, img_width, img_channels = 272, 480, 3
latent_dim = 12
def load_dataset(img_height, img_width):
    dataset_files = []
    for file in os.listdir("../Trainings_Data/vision_trainings_data"):
        dataset_files.append("../Trainings_Data/vision_trainings_data" + "/" + file)
        print(file)
    print(f'Total files: {len(dataset_files)}')

    dataset_files = dataset_files[46:49]
    #dataset_files = random.sample(dataset_files, 50)

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
        #image = tf.image.resize(image, [img_height, img_width])  # Resize the image to a fixed size.
        # image = image / 255.0  # Normalize the pixel values to the range 0-1.
        return image

    # Load the TFRecords file.
    full_dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = full_dataset

    # Map the input data to the parser and decoder functions.
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(lambda x: (decode_image(x['image']), {'output': x['pose']}))
    return dataset
def PCA_method_images(images):
    #images = np.array(images)
    #print(images)
    #print(images.shape)
    # # Convert the image to grayscale
    # _image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    #
    # # Flatten the image array into a 1D vector
    # flattened_image = _image.flatten()
    # print(f'flattened_image: {flattened_image.shape}')

    flat_images = []

    for image in images:
        # Assuming `flattened_image` has shape (num_samples, num_features) or (1, num_features)
        # Reshape for single image case:
        #flattened_image = images.reshape(1, -1)
        flattened_image = image.flatten()
        #print(flattened_image)

        flat_images.append(flattened_image)
        # Reshape for multiple images case:
        #flattened_images = np.vstack(flattened_image)  # Stack along sample axis

    flat_images = np.array(flat_images)
    print("flat_images.shape:", flat_images.shape)
    print("flat_images.T:", flat_images.T.shape)


    # Apply PCA to the flattened image
    pca = PCA(n_components=12)
    pca.fit(flat_images.T)

    # Access the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance_ratio)

    # Access the principal components
    principal_components = pca.components_
    print("Principal Components:", principal_components.T)
    print("principal_components.shape", principal_components.shape)

    # Transform the image using the learned PCA model
    reduced_images = pca.transform(flat_images.T)

    print("reduced_images.shape:", reduced_images.shape)

    # cv2.imwrite(f'image_reduction_methods_images/PCA_images/orginal_image_{i}.jpg', image*255.0)
    # cv2.imwrite(f'image_reduction_methods_images/PCA_images/PCA_image_{i}.jpg', PCA_image*255.0)


    # Transform the flattened image using the learned PCA model
    #reduced_images = pca.transform(flat_images)
    # Reshape the reduced image to (12,)
    #reduced_images = reduced_images.reshape(12, )

    return principal_components.T
def PCA_method(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(gray_image.shape)

    # Convert the image to a NumPy array
    image_array = np.array(gray_image)

    # Flatten the image array into a 1D vector
    flattened_image = image_array.flatten()
    # Reshape the flattened image to (1, num_features)
    flattened_image = flattened_image.reshape(1, -1)
    print(flattened_image.shape)

    # Apply PCA to the flattened image
    pca = PCA(n_components=1)
    reduced_vector = pca.fit_transform(flattened_image)

    return reduced_vector

class Autoencoder(Model):
    def __init__(self, latent_dim, img_height, img_width, img_channels):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            # Flatten(),
            # Dense(latent_dim, activation='relu'),
            Input(shape=(img_height, img_width, img_channels)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Flatten(),
            Dense(latent_dim, activation='linear'),
        ])

        self.decoder = tf.keras.Sequential([
            # Dense(784, activation='sigmoid'),
            # Reshape((28, 28))
            Dense(34 * 60 * 128),
            Reshape((34, 60, 128)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(img_channels, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(train_x):
    # Reshape the input images to a vector
    #input_dim = img_height * img_width * img_channels

    # # Define the architecture of the autoencoder
    # input_img = Input(shape=(img_height, img_width, img_channels))
    # # Encoder
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Flatten()(x)
    # encoded = Dense(20, activation='linear')(x)
    #
    # # Decoder
    # x = Dense(34*60*128)(encoded)
    # x = Reshape((34, 60, 128))(x)  # Reshape the data
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Conv2D(img_channels, (3, 3), activation='sigmoid', padding='same')(x)
    #
    # # Create the autoencoder model
    # autoencoder = Model(input_img, decoded)

    autoencoder = Autoencoder(latent_dim, img_height, img_width, img_channels)

    # Define the learning rate
    learning_rate = 0.00001
    # Create the Adam optimizer with the learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    autoencoder.compile(optimizer=optimizer, loss='mse')

    #autoencoder.build((img_height, img_width, img_channels))
    #autoencoder.summary()

    # Train the autoencoder
    autoencoder.fit(train_x, train_x,
                    epochs=50,
                    batch_size=2,
                    shuffle=True,)

    print("Saved model: AE_checkpoints/AE_model_ckp")
    autoencoder.save_weights('AE_checkpoints/AE_model_ckp')
    encoder_model = autoencoder.encoder
    encoder_model.save('AE_checkpoints/encoder_model.h5')

    # Save the encoder model
    #encoder_model = Model(input_img, encoded)
    #encoder_model.save('encoder_model.h5')
    #test_image = train_x[0]
    #cv2.imwrite(f'image_reduction_methods_images/autoencoder_images/test_images_0.jpg', test_image*255.0)
    # # Generate reconstructed images
    #reconstructed_images = autoencoder.predict(train_x)
    #encoded_imgs = autoencoder.encoder(train_x).numpy()

    #encoded_imgs = autoencoder.encoder(train_x)
    #print(encoded_imgs[0])

def evaluate_AE_model(train_x):
    autoencoder = Autoencoder(latent_dim, img_height, img_width, img_channels)

    encoder = tf.keras.models.load_model('AE_checkpoints/encoder_model.h5')

    print("Loaded model: AE_checkpoints/AE_model_ckp")
    autoencoder.load_weights('AE_checkpoints/AE_model_ckp')

    # for image in train_x:
    #     encoded_imgs = autoencoder.encoder(image.reshape(-1, img_height, img_width, img_channels))
    #     print("eval: ", encoded_imgs)
    #
    #     decoded_imgs = autoencoder.decoder(encoded_imgs)
    #     #print(decoded_imgs[0])
    #
    #     cv2.imwrite(f'image_reduction_methods_images/autoencoder_images/img_test2.jpg', decoded_imgs[0].numpy()*255.0)
    #     break

    for i, (image) in enumerate(train_x):
        if i <= 75:
            encoded_imgs = autoencoder.encoder(image.reshape(-1, img_height, img_width, img_channels))
            state_vector = encoder.predict(image.reshape(-1, img_height, img_width, img_channels))
            print(f"eval: {encoded_imgs[0]} vs. {state_vector}")

            decoded_imgs = autoencoder.decoder(encoded_imgs)
            #print(decoded_imgs[0])
            img = image*255.0
            img = img.reshape(img_height, img_width, img_channels)
            reconstructed_image = decoded_imgs[0].numpy()*255.0
            reconstructed_image = reconstructed_image.reshape(img_height, img_width, img_channels)

            #print(img.shape)
            #print(reconstructed_image.shape)

            # Concatenate the images horizontally
            concatenated_image = np.hstack((reconstructed_image, img))

            #cv2.imwrite(f'image_reduction_methods_images/autoencoder_images/AE_train_images/img_AE_{i}.jpg', reconstructed_image)
            #cv2.imwrite(f'image_reduction_methods_images/autoencoder_images/AE_train_images/img_{i}.jpg', img)
            cv.imwrite(f'image_reduction_methods_images/autoencoder_images/AE_train_images/img_concat_{i}.jpg', concatenated_image)
        else:
            break


# Initiated the segmentation once!
print(f'Running with Segmentation')
segmentation_model = SegmentationModelV2()

def create_pickle_dataset():
    dataset = load_dataset(img_height, img_width)
    train_x = []
    for i, (image, pose) in enumerate(dataset):

        img = image.numpy()*255.
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = cv.resize(img, (1920, 1080))
        #gray_image = cv.cvtColor(image.numpy(), cv.COLOR_BGR2GRAY)

        segmented_image = segmentation_model.generate_masks(img)
        #segmented_image = cv.cvtColor(segmented_image, cv.COLOR_RGB2GRAY)
        segmented_image = cv.resize(segmented_image, (img_width, img_height))

        print(f"Segmenting image {i}, shape=({segmented_image.shape}), {np.max(segmented_image/255.)}")
        train_x.append(segmented_image/255.)

        cv.imwrite(f'vision_figures/Segmented/Seg-{i}.jpg', segmented_image)

    train_x = np.array(train_x)

    with open('MT_datset.pkl', 'wb') as f:  # open a text file
        pickle.dump(train_x, f) # serialize the list



create_pickle_dataset()

#with open('MT_datset.pkl', 'rb') as f:
#    train_x = pickle.load(f) # deserialize using load()

#train(train_x)
#evaluate_AE_model(train_x)



