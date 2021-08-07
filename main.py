import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from data import *
#from xception_unet import DeepLabV3Plus
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

data_dir = 'base_dir'

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#Data generator train args
data_gen_args = dict(rotation_range=180,
                    horizontal_flip=True, fill_mode='nearest')

#Data Generator Val args
data_gen_args_val = dict(fill_mode='nearest')


batch_size = 8 
#Training Generator
myGene = trainGenerator(batch_size,data_dir,'train_images','train_labels',data_gen_args_val,save_to_dir = None)

#Val Generator
myGeneval = trainGenerator(batch_size,data_dir,'val_images','val_labels',data_gen_args_val,save_to_dir = None)

#defining model
model = sm.Unet('efficientnetb4', input_shape=(512,512,3), classes=1, activation='sigmoid', encoder_weights='imagenet')
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-3), loss=sm.losses.DiceLoss(),metrics=[sm.metrics.iou_score,dice_coef, 'accuracy'])

#saving checkpoint
model_checkpoint = ModelCheckpoint('segmentation.h5', monitor='val_loss',verbose=1, save_best_only=True) 

#Training the model
model.fit_generator(myGene,validation_data = myGeneval, epochs=100, callbacks=[model_checkpoint],steps_per_epoch=int((2075*2)//batch_size),validation_steps=(519//batch_size))
