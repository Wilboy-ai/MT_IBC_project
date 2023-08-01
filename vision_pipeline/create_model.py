#import os
#import tensorflow as tf
from tensorflow import keras
from keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import categorical_crossentropy
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def create_AlexNet_mine(img_height, img_width, img_channel):
    dropout_rate = 0.5

    model = keras.Sequential([
        layers.Input((img_height, img_width, img_channel)),
        layers.Conv2D(96, (11, 11), padding='same'),
        layers.MaxPool2D((3, 3)),
        layers.Conv2D(256, (5, 5), padding='same'),
        layers.MaxPool2D((3, 3)),
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.MaxPool2D((3, 3)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(7, activation='linear'),
    ])

    return model


def create_AlexNet(img_height, img_width, img_channel):
    inputs = keras.Input(shape=(img_height, img_width, img_channel))

    # Convolutional layers.
    x = keras.layers.Conv2D(96, 11, strides=4, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(3, strides=2)(x)
    x = keras.layers.Conv2D(256, 5, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2)(x)
    x = keras.layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2)(x)

    # Flatten and fully connected layers.
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    # modified to have name 'output'
    #outputs = keras.layers.Dense(3, name='output')(x)
    outputs = keras.layers.Dense(7, activation='linear', name='output')(x)

    return keras.Model(inputs=inputs, outputs=outputs)



def VGG16(img_height, img_width, img_channel):
    inputs = keras.Input(shape=(img_height, img_width, img_channel))

    # Convolutional layers.
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(4096, activation='relu')(x)

    outputs = keras.layers.Dense(7, activation='linear', name='output')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_MT_Model(img_height, img_width):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    # Convolutional layers.
    x = keras.layers.Conv2D(64, (7, 7), (2, 2), activation='relu')(inputs)
    x = keras.layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = keras.layers.Conv2D(32, (5, 5), activation='relu')(x)
    # Flatten and fully connected layers.
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(20, activation='relu')(x)
    x = keras.layers.Dense(10, activation='relu')(x)
    outputs = keras.layers.Dense(7, activation='linear', name='output')(x)  # modified to have name 'output'
    return keras.Model(inputs=inputs, outputs=outputs)

def create_MT_Model_V2(img_height, img_width):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(units=64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(units=7, activation='linear', name='output')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_backbone_model_ResNet(img_height, img_width):
    # load pre-trained ResNet50 model without top classification layer
    base_model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
    # freeze base model layers
    base_model.trainable = False

    # add new regression layers
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    #x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    #x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    output = keras.layers.Dense(7, activation='linear', name='output')(x)

    # create new model
    model = keras.Model(inputs=base_model.input, outputs=output)

    return model

def create_backbone_model_EfficientNet(img_height, img_width):
    # Create an instance of the EfficientNetB0 model with pre-trained weights from ImageNet
    base_model = keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # freeze base model layers
    base_model.trainable = False

    # add new regression layers
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    output = keras.layers.Dense(7, activation='linear', name='output')(x)

    # create new model
    model = keras.Model(inputs=base_model.input, outputs=output)

    return model


def create_RESNET(img_height, img_width, num_residual_blocks):
    # Input layer
    #inputs = tf.keras.Input(shape=input_shape)
    inputs = keras.Input(shape=(img_height, img_width, 3))

    phase_size = 64

    # Initial convolutional layer
    x = layers.Conv2D(phase_size, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)


    for _ in range(num_residual_blocks):
        # Sub-block 1
        residual = x
        x = layers.Conv2D(phase_size, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Sub-block 2
        x = layers.Conv2D(phase_size, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])  # Residual connection
        x = layers.Activation('relu')(x)


    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    #outputs = layers.Dense(7, activation='linear')(x)
    output = keras.layers.Dense(7, activation='linear', name='output')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=output)

    return model


def create_split_model(img_height, img_width):
    # First sub-model for the first three values
    input_layer = keras.layers.Input(shape=(img_height, img_width, 3))
    # First sub-model for the first three values
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    output_first = keras.layers.Dense(3, activation='linear', name='output_first')(x1)

    # Second sub-model for the last four values
    x2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    output_second = keras.layers.Dense(4, activation='linear', name='output_second')(x2)

    # Concatenate the outputs of both sub-models
    merged_output = keras.layers.Concatenate(name='output')([output_first, output_second])

    # Create the final model
    model = keras.models.Model(inputs=input_layer, outputs=merged_output)

    return model




