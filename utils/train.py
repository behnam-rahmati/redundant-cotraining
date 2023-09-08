#!/usr/bin/env python2.7

import pydicom, cv2, re, math, shutil
import os, fnmatch, sys
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from itertools import izip
from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
from U_net import *
# from dense_net import build_FC_DenseNet
# print(tf.__version__)


seed = 1234
np.random.seed(seed)

# directory initilizations
ROOT_PATH = "C:\\Users\\user1\\data"

# save your randomly selected contours in this path 
#also save the indices that you selected for random selections- load the random indices in the redundant_CT.py line 247
TRAIN_CONTOUR_PATH = "C:\\Users\\user1\\OneDrive\\Desktop\\backup\\backup"
TRAIN_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_training')
VAL_CONTOUR_PATH = "C:\\Users\\user1\\data\\val_GT"
VAL_IMG_PATH = os.path.join(ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\user1\\data\\ContoursPart1\\ContoursPart1\\OnlineDataContours"
ONLINE_IMG_PATH = os.path.join(ROOT_PATH,
                   'challenge_online')
TEST_CONTOUR_PATH = "C:\\Users\\user1\\OneDrive\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_test')

                        

# loss functions and evaluation metrics

def dice_coef(y_true, y_pred, smooth=0.0):
    #print(np.array(y_true).shape)
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
    
def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

# each data(directory path for the ground truth, slice number,etc.)is represented as an object of a class
class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))
        self.slice_no =  math.floor(self.img_no/20) if self.img_no%20 !=0 else math.floor(self.img_no/20)-1
        self.ED_flag = True if ((self.img_no%20) < 10 and (self.img_no % 20) !=0) else False
        self.is_weak = 0
   
    
    def __str__(self):
        return 'Contour for case %s, image %d' % (self.case, self.img_no)
    
    __repr__ = __str__
    
# returns image and GT(ground truth) mask of a specific data with its data path 
def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8') # shape is 256, 256
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask

# extract all the contours (including their data path) inside a directory
def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)       
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)    
    return contours
    
    
# extracts the images and mask of a set of contours. read_contour which extracts images and masks of one specific contour is used here 
def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(list(contours))))
    print(len(contours))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask        
    return images, masks


if __name__== '__main__':

    contour_type = sys.argv[1] # 'i' for endocardium, 'm' for myocardium , and 'o' for epicardium
    print(contour_type)
    # uncomment this line
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 100
    
    # if you are using U-Net uncomment next line
    #crop_size = 128
    
    print('Mapping ground truth '+contour_type+' contours to images in train...')

    #train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    
    # extract the test contours
    test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
    test_ctrs = test_ctrs[0:len(test_ctrs)//10]
    
    # extract the train contours
    train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))
    print('Done mapping training set')
    
    ## if you are performing validation on a part of train data uncomment this part
    ## 0.1 of the data for dev
    #split = int(0*len(a))
    #train_ctrs=train_ctrs[split:]
    #print(len(a))
    
    print('\nBuilding Train dataset ...')
    # extract all the train images and masks from the list of contours
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                            crop_size=crop_size)
                                                

    # extract all the test images and masks from the list of test contours
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(test_ctrs,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    
    
    # use this to load equal random initizations in your redundant networks (discussed in ablation studies)
    
    """
    #randomly initialized weights- if you want to change it create your own random initializations
    weights_equal = 'C:\\Users\\user1\\cardiac-segmentation-master\\cardiac-segmentation-master\\my_initial_weights.h5'

        if (phase == 'm'):
            mask_train = mask_train_myo
            mask_dev = mask_val_myo
            model = fcn_model(input_shape, num_classes, weights=weights_equal)
        elif (phase =='i'):
            mask_train = mask_train_endo
            mask_dev = mask_val_endo
            model = fcn_model(input_shape, num_classes, weights=weights_equal)
            
        elif (phase == 'o'):
            mask_train = mask_train_epi
            mask_dev = mask_val_epi
            model = fcn_model(input_shape, num_classes, weights=weights_equal) """   
    
    # choose your model (FCN or U-net for SSL)
    model = fcn_model(input_shape, num_classes, weights=None)
    

    
    #model = unet(input_size = input_shape, pretrained_weights=None)    

    #model = build_FC_DenseNet(model_version='fcdn56', nb_classes=2, final_softmax=True, 
                                    # input_shape=input_shape, dropout_rate=0.2, 
                                    # data_format='channels_last')

    #RMSprop(lr=0.001)
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss= dice_coef_loss,
                  metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)
                  
    #model.compile(optimizer=Adam(lr=0.0001),
                        # loss='sparse_categorical_crossentropy',
                        # metrics=[sparse_categorical_accuracy, mean_iou(num_classes=nb_classes)])
    #print(model.summary())
    
    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_train)/mini_batch_size)):
            img, mask = next(train_generator)
            # train
            res = model.train_on_batch(img, mask, sample_weight = 1)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        # validate
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['dataset-name', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('realtime'):
            os.makedirs('realtime')
        save_path = os.path.join('realtime', save_file)
        #print(save_path)
        model.save_weights(save_path)



