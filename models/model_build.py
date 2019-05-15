from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers


def build_vgg16_single_input(input_shape=(256, 256, 3)):
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = models.Sequential()

    model.add(vgg_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    vgg_base.trainable = False

    return model
