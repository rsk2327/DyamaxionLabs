import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model,save_model
from keras.callbacks import ModelCheckpoint
from matplotlib.patches import Rectangle
import os
from scipy.misc import imsave
from tqdm import tqdm

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from math import sqrt
from keras.callbacks import History 
from keras.optimizers import Adam, SGD
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, merge
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.layers import BatchNormalization
from keras.models import Model
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
import keras.backend as K
from math import sqrt
from keras.callbacks import History 
import gc
from math import log

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)





def train_test_split(x,ratio = 0.1,imgSize=(256,256)):
    
    np.random.shuffle(x)
    
    testSize = int(ratio*len(x))
    trainSize = len(x) - testSize
    
    trainx = []
    trainy = []
    testx = []
    testy = []
    
    for i in range(trainSize):
        trainx.append(x[i][0])
        trainy.append(x[i][1][:,:,0].reshape((imgSize[0],imgSize[1],1)))
        
    for i in range(trainSize,len(x)):
        testx.append(x[i][0])
        testy.append(x[i][1][:,:,0].reshape((imgSize[0],imgSize[1],1)))
    
    #Making masks into binary
    trainy = np.array(trainy)/255.0
    testy = np.array(testy)/255.0
    
    return(np.array(trainx),trainy,np.array(testx),testy)
    



def getCrossEntropy(mask,pred):
    mask = list(mask.reshape(-1))
    pred = list(pred.reshape(-1))
    
    entropy = 0.0
    
    for i in range(len(mask)):
        if mask[i] ==1.0:
            entropy += -log(pred[i] + 0.000001)
        else:
            entropy += -log(1-pred[i] + 0.000001)
            
    return(entropy/len(mask))


def getDice(mask,pred):
    
    mask = mask.reshape(-1)
    pred = pred.reshape(-1)
    
    
    intersect = np.sum(mask*pred)
    dice = (2.0*intersect + 1.0)/(np.sum(mask*mask) + np.sum(pred*pred) + 1.0)
    
    return dice



def getAccuracy(mask,pred):
    
    correctPred =0.0
    
    pred = list(pred.reshape(256*256,))
    mask = list(mask.reshape(256*256,))
    pred = [1.0 if x>=0.5 else 0.0 for x in pred]
    
    for i in range(256*256):
        if pred[i]==mask[i]:
            correctPred += 1
    
    return(correctPred/(256*256))
        

def toBinary(pred, threshold=0.5):
    
    pred = list(pred.reshape(256*256,))
    pred = np.array([1.0 if x>=threshold else 0.0 for x in pred])
    pred = pred.reshape((256,256,1))
    
    return(pred)


def invertMask(mask):
    
    maskShape = mask.shape
    mask = list(mask.reshape(-1))
    newmask = np.array([1.0 if x<0.5 else 0.0 for x in mask])
    
    newmask = newmask.reshape(maskShape)
    return newmask
    