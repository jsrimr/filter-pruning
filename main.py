# %%
import shutil
from pathlib import Path

import numpy as np

from tensorflow.keras import layers, models, regularizers, datasets, utils, callbacks, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# %%
import tensorflow as tf
from utils import ModelParametersCallback, ModelComplexityCallback
from model import resnet_v1, resnet_v2, resnet_mbconv, resnet_mbconv_v2



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
from pruning import KMeansFilterPruning, L1FilterPruning
from keras.datasets import cifar10
import keras
from jisup_resnet import ResNet

from tensorflow.keras.applications import MobileNet, MobileNetV3Small

if __name__ == "__main__":
    FIRST_TRAIN_EPOCHS = 2
    BATCH_SIZE = 128

    TRAIN_LOGS_FOLDER_PATH = Path("./train_logs")
    if TRAIN_LOGS_FOLDER_PATH.is_dir():
        shutil.rmtree(str(TRAIN_LOGS_FOLDER_PATH))
    TRAIN_LOGS_FOLDER_PATH.mkdir(exist_ok=True)

    model = ResNet()
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["accuracy"])
    # model = MobileNetV3Small(input_shape=(32,32,3), classes=10, weights=None)
    # model = MobileNet(input_shape=(32,32,3), classes=10, weights=None)
    # model = resnet_v1(input_shape=(32, 32, 3), depth=20, num_classes=10)
    # model = resnet_v2(input_shape=(32, 32, 3), depth=20, num_classes=10)
    # model = resnet_mbconv(input_shape=(32, 32, 3), depth=20, num_classes=10)
    # model = resnet_mbconv_v2(input_shape=(32, 32, 3), depth=20, num_classes=10)
    # compile_model(model)

    # print(model.summary())
    # raise "end"

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
    tensorboard_callback = callbacks.TensorBoard(log_dir=str(TRAIN_LOGS_FOLDER_PATH))  # profile_batch = '100,120'
    model_complexity_param = ModelParametersCallback(TRAIN_LOGS_FOLDER_PATH, verbose=1)
    # model_complexity_flops = ModelComplexityCallback(TRAIN_LOGS_FOLDER_PATH, verbose=1)
    model_checkpoint_callback = callbacks.ModelCheckpoint(str(TRAIN_LOGS_FOLDER_PATH) + "/model_{epoch:02d}.h5",
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        verbose=1)

    callbacks = [tensorboard_callback, model_checkpoint_callback, model_complexity_param ]

    # Train model
    # model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE),
    #                     epochs=FIRST_TRAIN_EPOCHS,
    #                     validation_data=(x_test, y_test),
    #                     callbacks=callbacks,
    #                     steps_per_epoch=STEPS_PER_EPOCH)

    # import keras
    # model = keras.models.load_model('Resnet_v1_train_logs/model_01.h5')
    # pruning = KMeansFilterPruning(0.9,
    #                                     compile_model,
    #                                     finetune_model,
    #                                     nb_finetune_epochs=1,
    #                                     maximum_pruning_percent=0.85,
    #                                     maximum_prune_iterations=3,
    #                                     nb_trained_for_epochs=FIRST_TRAIN_EPOCHS)

    pruning = L1FilterPruning(0.9,
                                        compile_model,
                                        finetune_model,
                                        nb_finetune_epochs=1,
                                        maximum_pruning_percent=0.85,
                                        maximum_prune_iterations=3,
                                        nb_trained_for_epochs=FIRST_TRAIN_EPOCHS)
    model, last_epoch_number = pruning.run_pruning(model)

