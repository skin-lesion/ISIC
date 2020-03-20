import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Check if GPU is available
# K.tensorflow_backend._get_available_gpus()

# The paths for the training and validation images
train_path = 'base_dirtrain_dir'
valid_path = 'base_dirval_dir'
test_path = 'base_dirtest_dir'
# Declare a few useful values
num_train_samples = 38704
num_val_samples = 1002
train_batch_size = 8
val_batch_size = 8
image_size = 224
import tensorflow
# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples  train_batch_size)
val_steps = np.ceil(num_val_samples  val_batch_size)

# Create a MobileNet model
from classification_models.tfkeras import Classifiers
resnet, preprocess_input = Classifiers.get('xception')
import tensorflow
base_model = resnet(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tensorflow.keras.layers.Dense(7, activation='softmax')(x)
model = tensorflow.keras.models.Model(inputs=[base_model.input], outputs=[output])
# See a summary of the new layers in the model
model.summary()

# Set up generators
train_batches = ImageDataGenerator(
    preprocessing_function= 
        tensorflow.keras.applications.xception.preprocess_input).flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size)

valid_batches = ImageDataGenerator(
    preprocessing_function= 
        tensorflow.keras.applications.xception.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size)


# Train the model
# Define Top2 and Top3 Accuracy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred)
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred)
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Compile the model
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

# Add weights to make the model more sensitive to melanoma

# Declare the filepath for the saved model
filepath = model_xception_crop.h5

# Declare a checkpoint to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                              verbose=1, mode='min', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

# Fit the model
history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=30,
                              verbose=1,
                              callbacks=callbacks_list)