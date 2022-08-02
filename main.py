# %%
import shutil
from pathlib import Path

import numpy as np

from tensorflow.keras import layers, models, regularizers, datasets, utils, callbacks, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# %%
import tensorflow as tf
from utils import ModelParametersCallback, ModelComplexityCallback

# %%


# %%
def _resnet_layer(inputs,
                  num_filters=16,
                  kernel_size=3,
                  strides=1,
                  activation='relu',
                  batch_normalization=True,
                  conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x



# %%
def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = _resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = _resnet_layer(inputs=x,
                              num_filters=num_filters,
                              strides=strides)
            y = _resnet_layer(inputs=y,
                              num_filters=num_filters,
                              activation="relu")
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = _resnet_layer(inputs=x,
                                  num_filters=num_filters,
                                  kernel_size=1,
                                  strides=strides,
                                  activation="relu",
                                  batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(y)

    # Instantiate model.
    model = models.Model(inputs=inputs, outputs=outputs)
    return model





# %%
def compile_model(my_model):
    my_model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["accuracy"])

# %%
# Prune the model
def finetune_model(my_model, initial_epoch, finetune_epochs):
    my_model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE),
                           epochs=finetune_epochs,
                           validation_data=(x_test, y_test),
                           callbacks=callbacks,
                           initial_epoch=initial_epoch,
                           verbose=1,
                           steps_per_epoch=STEPS_PER_EPOCH)

# %%
from pruning import KMeansFilterPruning
from keras.datasets import cifar10
import keras


if __name__ == "__main__":
    FIRST_TRAIN_EPOCHS = 20
    BATCH_SIZE = 32

    TRAIN_LOGS_FOLDER_PATH = Path("./train_logs")
# if TRAIN_LOGS_FOLDER_PATH.is_dir():
#     shutil.rmtree(str(TRAIN_LOGS_FOLDER_PATH))
    TRAIN_LOGS_FOLDER_PATH.mkdir(exist_ok=True)

    model = resnet_v1(input_shape=(32, 32, 3), depth=20, num_classes=10)
    compile_model(model)

    # Loading data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Data Transform
    x_train = x_train.astype(np.float32) / 255.0
    y_train = utils.to_categorical(y_train)
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean

    x_test = x_test.astype(np.float32) / 255.0
    y_test = utils.to_categorical(y_test)
    x_test -= x_train_mean

    print("Train shape: X {0}, y: {1}".format(x_train.shape, y_train.shape))
    print("Test shape: X {0}, y: {1}".format(x_test.shape, y_test.shape))
    STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

    # Data Augmentation with Data Generator
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)

    # Create callbacks
    tensorboard_callback = callbacks.TensorBoard(log_dir=str(TRAIN_LOGS_FOLDER_PATH))
    model_complexity_param = ModelParametersCallback(TRAIN_LOGS_FOLDER_PATH, verbose=1)
    model_complexity_flops = ModelComplexityCallback(TRAIN_LOGS_FOLDER_PATH, tf.compat.v1.get_default_session(), verbose=1)
    model_checkpoint_callback = callbacks.ModelCheckpoint(str(TRAIN_LOGS_FOLDER_PATH) + "/model_{epoch:02d}.h5",
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        verbose=1)
    callbacks = [tensorboard_callback, model_complexity_param, model_checkpoint_callback]

    # Train model
    # model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE),
    #                     epochs=FIRST_TRAIN_EPOCHS,
    #                     validation_data=(x_test, y_test),
    #                     callbacks=callbacks,
    #                     steps_per_epoch=STEPS_PER_EPOCH)

    # import keras
    model = keras.models.load_model('archive/model_20.h5')
    pruning = KMeansFilterPruning(0.9,
                                        compile_model,
                                        finetune_model,
                                        nb_finetune_epochs=1,
                                        maximum_pruning_percent=0.85,
                                        maximum_prune_iterations=10,
                                        nb_trained_for_epochs=FIRST_TRAIN_EPOCHS)
    model, last_epoch_number = pruning.run_pruning(model)

