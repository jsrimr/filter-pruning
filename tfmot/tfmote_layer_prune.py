import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)


def setup_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=input_shape),
        tf.keras.layers.Flatten()
    ])
    return model


def setup_pretrained_weights():
    model = setup_model()

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train)

    _, pretrained_weights = tempfile.mkstemp('.tf')

    model.save_weights(pretrained_weights)

    return pretrained_weights


def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)


setup_model()
pretrained_weights = setup_pretrained_weights()


base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended.

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_pruning.summary()


# Create a base model
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy

# Helper function uses `prune_low_magnitude` to make only the 
# Dense layers train with pruning.
def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

model_for_pruning.summary()