# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:32:14 2019

@author: coeuser02
"""

import numpy as np
from glob import glob
import ntpath
import nibabel
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np



import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras



from keras.losses import binary_crossentropy
from keras import backend as K

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization,AveragePooling2D
from keras.layers import Conv2D, Concatenate, MaxPooling2D,Add,SeparableConv2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.regularizers import l2

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import*
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/COE1/Desktop/skin/train',
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'C:/Users/COE1/Desktop/skin/validation',
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical')

#
#
#def tversky(y_true, y_pred):
#    y_true_pos = K.flatten(y_true)
#    y_pred_pos = K.flatten(y_pred)
#    true_pos = K.sum(y_true_pos * y_pred_pos)
#    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#    alpha = 0.7
#    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
#
#def tversky_loss(y_true, y_pred):
#    return 1 - tversky(y_true,y_pred)
#
#
#
#
#def residual_block(x,channel,dilation_rate):
#    #    x_shape=x.shape()
#    conv1 = tf.compat.v1.keras.layers.Conv2D(channel, (3, 3), activation='relu',dilation_rate=dilation_rate, padding='same')(x)
#    conv1 = tf.compat.v1.keras.layers.BatchNormalization()(conv1)
#    conv2 = tf.compat.v1.keras.layers.Conv2D(channel, (3, 3), activation='relu',dilation_rate=dilation_rate, padding='same')(conv1)
#    conv2 = tf.compat.v1.keras.layers.BatchNormalization()(conv2)
#    x=tf.compat.v1.keras.layers.Add()([conv2,x])
#    return x
#
#
#def residual_decoder(x1,channel1):
#    #    x_shape=x.shape()
#    #channel_half = int (channel1/2)
#    conv1 = tf.compat.v1.keras.layers.Conv2D(channel1, (3, 3), activation='relu', padding='same')(x1)
#    conv1 = tf.compat.v1.keras.layers.BatchNormalization()(conv1)
#    conv2 = tf.compat.v1.keras.layers.Conv2D(channel1, (3, 3), activation='relu', padding='same')(conv1)
#    conv2 = tf.compat.v1.keras.layers.BatchNormalization()(conv2)
#    conv2_d = tf.compat.v1.keras.layers.Conv2D(channel1, (1, 1), activation='relu', padding='same')(x1)
#    x=tf.compat.v1.keras.layers.Add()([conv2,conv2_d])
#    return x
#
#def res_segment_c (input_shape):
#    input_img = tf.compat.v1.keras.layers.Input(input_shape)
#
#    ## (,,64)
#    conv1 = tf.compat.v1.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
#    conv1 = tf.compat.v1.keras.layers.BatchNormalization()(conv1)
#    print(conv1.shape)
#    ## Residual Block Part
#    residual1=residual_block(conv1,64,dilation_rate=(1,1))
#    print(residual1.shape)
#    ## Max pooling or convulational downsampling
#    pool1 = tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual1)
#
#    ## (,,128)
#    conv2 = tf.compat.v1.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
#    conv2 = tf.compat.v1.keras.layers.BatchNormalization()(conv2)
#    ## Residual Block Part
#    residual2=residual_block(conv2,128,dilation_rate=(1,1))
#    ## Max pooling or convulational downsampling
#    pool2 = tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual2)
#
#    ## (,,256)
#    conv3 = tf.compat.v1.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
#    conv3 = tf.compat.v1.keras.layers.BatchNormalization()(conv3)
#    ## Residual Block Part
#    residual3=residual_block(conv3,256,dilation_rate=(1,1))
#    print(residual3.shape)
#    ## Max pooling or convulational downsampling
#    pool3 = tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual3)
#
#    ## (,,512)
#    conv4 = tf.compat.v1.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
#    conv4 = tf.compat.v1.keras.layers.BatchNormalization()(conv4)
#    ## Residual Block Part
#    residual4=residual_block(conv4,512,dilation_rate=(1,1))
#    print(residual4.shape)
#    ## Max pooling or convulational downsampling
#    pool4 = tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual4)
#
#    conv5 = tf.compat.v1.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
#    conv5 = tf.compat.v1.keras.layers.BatchNormalization()(conv5)
#    residual5=residual_block(conv5,1024,dilation_rate=(1,1))
#
#    ##decoder part
#    up6 = tf.compat.v1.keras.layers.UpSampling2D((2,2))(residual5)
#    print(up6.shape)
#    conv6 = tf.compat.v1.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
#    conv6 = tf.compat.v1.keras.layers.BatchNormalization()(conv6)
#    #conv6 = Add()([conv6, residual4])
#    conv6 = tf.compat.v1.keras.layers.concatenate([conv6,residual4], axis = 3)
#    residual6=residual_decoder(conv6,512)
#
#    up7 = tf.compat.v1.keras.layers.UpSampling2D((2,2))(residual6)
#    conv7 = tf.compat.v1.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
#    conv7 = tf.compat.v1.keras.layers.BatchNormalization()(conv7)
#    #conv7 = Add()([conv7, residual3])
#    conv7 = tf.compat.v1.keras.layers.concatenate([conv7,residual3], axis = 3)
#    residual7=residual_decoder(conv7,256)
#
#    up8 = tf.compat.v1.keras.layers.UpSampling2D((2,2))(residual7)
#    conv8 = tf.compat.v1.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
#    conv8 = tf.compat.v1.keras.layers.BatchNormalization()(conv8)
#    #conv8 = Add()([conv8, residual2])
#    conv8 = tf.compat.v1.keras.layers.concatenate([conv8,residual2], axis = 3)
#    residual8=residual_decoder(conv8,128)
#
#    up9 = tf.compat.v1.keras.layers.UpSampling2D((2,2))(residual8)
#    conv9 = tf.compat.v1.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
#    conv9 = tf.compat.v1.keras.layers.BatchNormalization()(conv9)
#    #conv9 = Add()([conv9, residual1])
#    conv9 = tf.compat.v1.keras.layers.concatenate([conv9,residual1], axis = 3)
#    residual9=residual_decoder(conv9,64)
#
#    #decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
#    decoded = tf.compat.v1.keras.layers.Conv2D(8, (1, 1), activation='sigmoid', padding='same')(residual9)
#    nis_final = tf.compat.v1.keras.Model(input_img,decoded)
#	
#	
#	
#	
#    return nis_final




def create_train_valid_data(name_data):
#    src_path="/home/user/Task3_Thoracic_OAR/Thoracic_OAR/*"
    src_path="/home/romil/Downloads/classes_1p19q/dataset_1p19q/"+name_data+"/*"
    patients_dir = glob(src_path)
    total_num_pat=len(patients_dir)
    return patients_dir, total_num_pat
    
def step_decay(epoch):
    res = 0.01
    if epoch > 10:
        res = 0.005
    if epoch > 20:
        res = 0.001
    print("learnrate: ", res, " epoch: ", epoch)
    return res   

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 99)
    x_min = np.percentile(x, 1)    
    x = (x - x_min) / (x_max - x_min)
#    x = x.clip(0, 1)
    x[:,:,2][x[:,:,2] > 0] = 1
    return x


def data_generator(batch_size,data):

    batch_idx = 0
    
    random_state = np.random.RandomState(1301)
    while True:
        batch_img_list = []
        batch_gt_list = []

        for index, each_pat in enumerate(data):
            pat_ori_file_path=denormalize(np.load(each_pat))
            pat_gt_file_path= int(each_pat.split('_')[-1].replace('.npy',''))
            pat_gt_file_path = tf.keras.utils.to_categorical(pat_gt_file_path,2)
            each_ori_img=pat_ori_file_path.reshape(1,256,256,3)
            batch_img_list.append(each_ori_img)
            batch_gt_list.append(pat_gt_file_path)
            batch_idx += 1
            if batch_idx >= batch_size:
                
                x = np.vstack(batch_img_list)
                y = np.vstack(batch_gt_list)
                yield x,y
                batch_img_list = []
                batch_gt_list = []
                batch_idx = 0
    
from classification_models.keras import Classifiers

            

def train(model_name):
    batch_size = 8
    
#    train_pat,train_steps= create_train_valid_data('train_set')
#    valid_pat,validation_steps= create_train_valid_data('val_set')
#
#    train_gen = data_generator(batch_size,train_pat)
#    validation_gen = data_generator(batch_size, valid_pat)


    
    
#    print(train_steps)
#    print(validation_steps)

    learnrate_scheduler = LearningRateScheduler(step_decay)
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = ResNet18(input_shape=(512, 512, 3), weights=None, classes = 8)                
    model.summary()
#    model = res_segment_c((512,512,1))
#    model.summary()
#    model.load_weights('/home/suraj/tiger_weight/model_resnet_seg08-0.01770.0034-0.9873-0.9812-0.9986.hd5')
#    model.compile(loss=bce_dice_loss, optimizer="adam", metrics=[dice_coef, "accuracy"])
#    model.compile(optimizer=Adam(clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef, "acc"])
#    com_model=tf.compat.v1.keras.utils.multi_gpu_model(model,gpus=2)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #model.summary()

    checkpoint = ModelCheckpoint("skin_model.hd5", monitor='val_loss', verbose=1, save_best_only=True)
    history=model.fit_generator(train_generator,int(22817/batch_size),40,validation_data = validation_generator, validation_steps =int(2514/batch_size), verbose=1,class_weight="balanced",callbacks=[checkpoint])


if __name__ == "__main__":
    train(model_name="resnet_seg")



