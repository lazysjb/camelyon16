from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate, Dense, Dropout, BatchNormalization, Flatten, Input,
)


def build_vgg16_single_input(input_shape=(256, 256, 3)):
    """Corresponds to model_vgg_ver2 of Modelling.ipynb file"""
    # Single input VGG16 transfer learn
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = models.Sequential()

    model.add(vgg_base)
    model.add(Flatten())
    model.add(Dropout(rate=0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    vgg_base.trainable = False

    return model


def build_vgg16_double_input(input_shape=(256, 256, 3)):
    """Corresponds to model_10 of Modelling.ipynb file"""
    # Double input VGG16 transfer learn
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # define two sets of inputs
    zoom_1 = Input(shape=input_shape)
    zoom_2 = Input(shape=input_shape)

    # process zoom level 1 patch
    conv_1 = vgg_base(zoom_1)
    flatten_1 = Flatten()(conv_1)

    # process zoom level 2 patch
    conv_2 = vgg_base(zoom_2)
    flatten_2 = Flatten()(conv_2)

    # combine output of convolutional layers
    combined = concatenate([flatten_1, flatten_2])
    combined = BatchNormalization()(combined)
    combined = Dropout(rate=0.4)(combined)

    # fully connected layer after combined outputs
    z = Dense(128, activation="relu")(combined)
    z = BatchNormalization()(z)
    z = Dropout(rate=0.4)(z)
    z = Dense(1, activation="sigmoid")(z)

    vgg_base.trainable = False

    model = Model(inputs=[zoom_1, zoom_2], outputs=z)

    return model
