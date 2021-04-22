import os
import multiprocessing
import rasterio
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras as keras
# import keras
import pandas as pd
import boto3
import io
import json
from tensorflow_addons.metrics import F1Score, HammingLoss
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import random

    
    
def load_model(model_url,weights_url):

    s3 = boto3.resource('s3')
    model_filename = "model.h5"
    model_weights_filename = "model_weights.h5"

    #Download Model, Weights

    bucket = model_url.split("/")[2]
    model_key = "/".join(model_url.split("/")[3:])
    s3.Bucket(bucket).download_file(model_key, model_filename)
    weights_key = "/".join(weights_url.split("/")[3:])
    s3.Bucket(bucket).download_file(weights_key, model_weights_filename)


    model = tf.keras.models.load_model(model_filename)
    model.load_weights(model_weights_filename)

    return model 


def read_image_tf_out(window_arr):

    #create copy of bands list, remove ndvi band from copy 
    window_arr_no_ndvi = window_arr.copy()
    window_arr_no_ndvi = window_arr_no_ndvi[:-1] 
    tf_img_no_ndvi = tf.image.convert_image_dtype(window_arr_no_ndvi, tf.float32)
    
    ndvi_band = window_arr_no_ndvi[-1]
    tf_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
    
    tf_img = tf.concat([tf_img_no_ndvi,[tf_img_ndvi]],axis=0)
    tf_img = tf.transpose(tf_img,perm=[1, 2, 0])
    tf_img = tf.expand_dims(tf_img, axis=0)

    return tf_img

def read_image(window_arr):

    #create copy of bands list, remove ndvi band from copy 
    window_arr_no_ndvi = window_arr.copy()
    window_arr_no_ndvi = window_arr_no_ndvi[:-1].astype('float32')
    
    ndvi_band = window_arr_no_ndvi[-1].astype('float32')
    
    img = np.concatenate([window_arr_no_ndvi,[ndvi_band]],axis=0)
    img = np.transpose(img, (1, 2, 0))


    return np.array([img])



if __name__ == '__main__':

    bands=[2, 3, 4, 8,11,12,18]
    input_shape_RGBNIRSWR1SWR2NDVI = (100,100,len(bands))
                                        
                                        

