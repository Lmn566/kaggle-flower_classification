"""
convert_tfrecord REFERENCE : 
READ TFRECORDS FILES IN PYTORCH
https://medium.com/analytics-vidhya/how-to-read-tfrecords-files-in-pytorch-72763786743f
"""

import glob
import io
import os.path as osp

import numpy as np
from PIL import Image
import tensorflow as tf

def convert_tfrecord(files, mode='train'):
    if mode != 'test': # train, val
        id_list, class_list, image_list = list(), list(), list()
    
        feature_description = {
            'class': tf.io.FixedLenFeature(list(), tf.int64),
            'id': tf.io.FixedLenFeature(list(), tf.string),
            'image': tf.io.FixedLenFeature(list(), tf.string),
        }
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
    
        for i in files:
            image_dataset = tf.data.TFRecordDataset(i)
          
            image_dataset = image_dataset.map(_parse_image_function)
          
            ids = [str(id_features['id'].numpy())[2:-1] for id_features in image_dataset] 
            id_list += ids
          
            classes = [int(class_features['class'].numpy()) for class_features in image_dataset]
            class_list += classes
          
            images = [image_features['image'].numpy() for image_features in image_dataset]
            image_list += images

        return id_list, class_list, image_list


    elif mode=='test': # test
        id_list, class_list, image_list = list(), list(), list()
    
        feature_description = {
            'id': tf.io.FixedLenFeature(list(), tf.string),
            'image': tf.io.FixedLenFeature(list(), tf.string),
        }
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
    
        for i in files:
            image_dataset = tf.data.TFRecordDataset(i)
          
            image_dataset = image_dataset.map(_parse_image_function)
          
            ids = [str(id_features['id'].numpy())[2:-1] for id_features in image_dataset] 
            id_list += ids
          
            images = [image_features['image'].numpy() for image_features in image_dataset]
            image_list += images

        return id_list, image_list
