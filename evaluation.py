from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow 
import numpy as np
from classification_models.tfkeras import Classifiers
resnet, preprocess_input = Classifiers.get('xception')
import itertools 
test_path = 'base_dir/test_dir'

num_test_samples = 1002
test_batch_size = 8
image_size = 224


test_batches = ImageDataGenerator(
    preprocessing_function= \
        tensorflow.keras.applications.xception.preprocess_input).flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=test_batch_size,
    shuffle=False)

base_model = resnet(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tensorflow.keras.layers.Dense(7, activation='softmax')(x)
model = tensorflow.keras.models.Model(inputs=[base_model.input], outputs=[output])
# See a summary of the new layers in the model
model.summary()
model.load_weights('model_mobilenetv2.h5')

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, steps=num_test_samples/test_batch_size)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

# Create a confusion matrix of the test images
test_labels = test_batches.classes

# Make predictions
predictions = model.predict_generator(test_batches, steps=num_test_samples/test_batch_size, verbose=1)

# Declare a function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels)