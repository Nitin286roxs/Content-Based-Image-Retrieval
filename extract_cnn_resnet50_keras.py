"""
Date: 8-Aug-2020
Author: Nitin Ashutosh <nitinashu1995@gmail.com>
"""
import numpy as np
from numpy import linalg as LA
#from PIL import Image, ImageFilter
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

class ResNet:
    '''
    Loading pretrained ResNet50 model and extracting feature vector from images
    '''
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model = None
    def load_network(self):
        '''
        Loading pretrained ResNet50.
        weights: 'imagenet'
        pooling: 'max' or 'avg'
        input_shape: (width, height, 3), width
        '''
        self.model = ResNet50(weights='imagenet', input_shape=(self.input_shape[0],\
            self.input_shape[1], self.input_shape[2]), pooling='max', include_top=False)
    def extract_feat(self, img_path):
        '''
        Use resnet50 model to extract features Output normalized feature vector
        '''
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
