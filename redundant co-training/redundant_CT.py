#!/usr/bin/env python2.7
from scipy import ndimage
import skimage.measure
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
from fcn_model import dice_coef
import re, sys, os, time
import shutil, cv2
import numpy as np
import matplotlib.pyplot as plt
from train import read_contour, map_all_contours, export_all_contours
from fcn_model import fcn_model
from helpers import reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
crop_size = 100
#if you are using U-Net uncomment next line
#crop_size = 128
seed = 1234
np.random.seed(seed)

# specify the directories of data

ROOT_PATH = "C:\\Users\\user1\\data"
VAL_CONTOUR_PATH = "C:\\Users\\user1\\data\\valGT"
VAL_IMG_PATH = os.path.join(ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\user1\\data\\online"
ONLINE_IMG_PATH = os.path.join(ROOT_PATH,
                   'challenge_online')

TRAIN_CONTOUR_PATH = "C:\\Users\\user1\\OneDrive\\Desktop\\backup\\backup"
TRAIN_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_training')
TEST_CONTOUR_PATH = "C:\\Users\\user1\\OneDrive\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_test')   
                        


# this function creates pseudo-labels for the unlabeled data based on our proposed SSL approach

def create_pseudo_labels(contours, data_path):

    crop_size = 100
    
    # if you are using U-Net uncomment the next line
    #crop_size = 128
    
    #load the unlabeled images
    images, _ = export_all_contours(contours, data_path, crop_size)
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2

    # load the weights of the trained redundant netwokrs (epicardium, endocardium, myocardium)
    # this networks should be trained seperately using train.py with supervised only method (no SSL)
    # the train data should be selected randomly and based on each specific experiment (20-30-60-120-1/20-1/10-1/5)
    # you should generate the random indices and train the networks using train.py before running this code

    weights_i = 'C:\\Users\\user1\\cardiac-segmentation-master\\cardiac-segmentation-master\\all_weights\\20i.h5'
    weights_m = 'C:\\Users\\user1\\cardiac-segmentation-master\\cardiac-segmentation-master\\all_weights\\20m2.h5'   
    weights_o = 'C:\\Users\\user1\\cardiac-segmentation-master\\cardiac-segmentation-master\\all_weights\\20o.h5'

    
    # define the redundant networks and load their weights obtained from only supervised training

    model1 = fcn_model(input_shape, num_classes, weights=weights_i) #endocardium
    model2 = fcn_model(input_shape, num_classes, weights=weights_m) #myocardium
    model3 = fcn_model(input_shape, num_classes, weights=weights_o) #epicardium
    
    # obtain the redundant models predictions on unlabeled data
    
    pred_masks1 = model1.predict(images, batch_size=32, verbose=1)
    pred_masks2 = model2.predict(images, batch_size=32, verbose=1)
    pred_masks3 = model3.predict(images, batch_size=32, verbose=1)
    
    pseudo_masks = []
    imgs_retrain=[]
    masks_retrain = []
    num = 0
    selected_ctr1=[]
    selected_ctr2=[]
    
    #thresholds for the reliability estimations (discussed in the supplementary matarial)
    thresh1 = 0.03
    thresh2=0.001
    
    # assigns Psudo-labels for each unlabeled image and for each primary task (epicardium, myocardium, endocardium)
    
    for idx, ctr in enumerate(contours):
    
        img, _ = read_contour(ctr, data_path)

        tmp1 = pred_masks1[idx] #endocardium
        tmp2 = pred_masks2[idx] #myocardium
        tmp3 = pred_masks3[idx] #epicardium
        
        pseudo_endo = np.copy(tmp1)
        pseudo_myo = np.copy(tmp2)
        pseudo_epi = np.copy(tmp3)
       
        # reliablity of the primary predictions (discussed in the supplementary materials)
        reliable_mask_endo = np.zeros_like(tmp1)
        reliable_tmp = np.logical_or(tmp1< thresh2 , tmp1>1- thresh2) 
        reliable_mask_endo[reliable_tmp] = 1
        
        reliable_mask_myo = np.zeros_like(tmp2)
        reliable_tmp = np.logical_or(tmp2< thresh2 , tmp2>1- thresh2) 
        reliable_mask_endo[reliable_tmp] = 1

        reliable_mask_epi = np.zeros_like(tmp3)
        reliable_tmp = np.logical_or(tmp3< thresh1 , tmp3>1- thresh1) 
        reliable_mask_endo[reliable_tmp] = 1        
        
        # reliability of the redundant predictions
        redundant_mask_endo = np.zeros_like(tmp1)
        redundant_tmp = (np.logical_or(tmp2<thresh1, tmp2>1-thresh1) &  np.logical_or(tmp3<thresh1 , tmp3>1-thresh1))
        redundant_mask_endo[redundant_tmp] = 1  

        redundant_mask_myo = np.zeros_like(tmp2)
        redundant_tmp = (np.logical_or(tmp1<thresh1, tmp1>1-thresh1) &  np.logical_or(tmp3<thresh1 , tmp3>1-thresh1))
        redundant_mask_endo[redundant_tmp] = 1 

        redundant_mask_endo = np.zeros_like(tmp3)
        redundant_tmp = (np.logical_or(tmp1<thresh1, tmp1>1-thresh1) &  np.logical_or(tmp2<thresh1 , tmp2>1-thresh1))
        redundant_mask_endo[redundant_tmp] = 1         
            
        # the next lines can be changed to _myo and _epi for myocardium and epicardium 
        primary_mask = reliable_mask_endo
        #primary_mask = reliable_mask_myo
        #primary_mask = reliable_mask_epi

        redundant_mask = redundant_mask_endo
        #redundant_mask = redundant_mask_myo
        #redundant_mask = redundant_mask_epi

        
        # if the primary predictions are unreliable but redundant predictions are reliable the pixels_to_change would be True
        pixels_to_change = (primary_mask ==0) & (redundant_mask==1)
        tmp3_mask = np.zeros_like(tmp3)

        # endocardium pseudo-labels- the default code will return this 
        # change line 106 and 110 if the primary task is myocardium or epicardium segmentation
        pseudo_endo [pixels_to_change] = ((tmp3 [pixels_to_change]) - tmp2 [pixels_to_change]) 
        pseudo_endo [pseudo_endo < 0.5] = 0
        pseudo_endo [pseudo_endo > 0.5] = 1 
        #pseudo_endo [pixels_to_change] = ((tmp3 [pixels_to_change]) - tmp2 [pixels_to_change]) 
        #the predictions can be thresholded before or after the co-training. To change that uncomment this line and comment line 137
        
        # myocardium pseudo-labels- lines 122 and 126 should be replaced with lines 123 and 127 if the primary task is myocardium segmentation
        pseudo_myo [pixels_to_change] = ((tmp3 [pixels_to_change]) - tmp1 [pixels_to_change]) 
        pseudo_myo [pseudo_myo < 0.5] = 0
        pseudo_myo [pseudo_myo > 0.5] = 1 
        #pseudo_myo [pixels_to_change] = ((tmp3 [pixels_to_change]) - tmp1 [pixels_to_change]) 
        #the predictions can be thresholded before or after the co-training. To change that uncomment this line and comment line 144
        

        # epicardium pseudo-labels- lines 122 and 126 should be replaced with lines 124 and 128 if the primary task is epicardium segmentation  
        pseudo_epi [pixels_to_change] = ((tmp1 [pixels_to_change]) + tmp2 [pixels_to_change]) 
        pseudo_epi [pseudo_epi < 0.5] = 0
        pseudo_epi [pseudo_epi > 0.5] = 1   
        #pseudo_epi [pixels_to_change] = ((tmp1 [pixels_to_change]) + tmp2 [pixels_to_change]) 
        #the predictions can be thresholded before or after the co-training. To change that uncomment this line and comment line 152
        

        # this part is used for our proposed uncertainty estimation in the ablation studies and discussions
        diff = abs(tmp3-tmp1-tmp2)
        diff[diff<0.5] = 0

        selected_ctr1.append(ctr)
        img = center_crop(img, crop_size=crop_size)    
        imgs_retrain.append(img)
        
        mask = center_crop(pseudo_endo, crop_size=crop_size)
        #mask = center_crop(pseudo_myo, crop_size=crop_size)
        #mask = center_crop(pseudo_epi, crop_size=crop_size)
        
        masks_retrain.append (mask)

       
       
        img = center_crop(img, crop_size=crop_size)    
  
    return np.array(imgs_retrain) , np.array(masks_retrain)
    
    
    
if __name__== '__main__':
    crop_size = 100
    
    #uncomment this line
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    # extract the train data (they should be defined by their names in text files in the directory initializations (line 19))
    contour_type = 'i' # should be changed to 'm' for myocardium and 'o' for epicardium
    endo_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type = 'i', shuffle=False))
    myo_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type = 'm', shuffle=False))
    epi_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type = 'o', shuffle=False))

    # this part is done because in one of the datasets the epicardium labels were available only in the ED phase
    img_no = [ctr.img_no for ctr in epi_ctrs]
    for ctr in endo_ctrs:
        if not(ctr.img_no in img_no):
            endo_ctrs.remove(ctr)

    img_epi, mask_epi = export_all_contours(epi_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)

    img_endo, mask_endo = export_all_contours(endo_ctrs,
                                            TRAIN_IMG_PATH,
                                                crop_size=crop_size)
                                                
    # myocardium mask                                            
    mask_myo = (np.array(mask_epi) - np.array(mask_endo)) 

    # test contours should be defined in the directory initializtions (line 19)
    test_ctrs_endo = list(map_all_contours(TEST_CONTOUR_PATH, contour_type = 'i', shuffle=False))
    test_ctrs_epi = list(map_all_contours(TEST_CONTOUR_PATH, contour_type = 'o', shuffle=False))

    # this part is done because in one of the datasets the epicardium labels were available only in the ED phase
    img_no = [ctr.img_no for ctr in test_ctrs_epi]
    for ctr in test_ctrs_endo:
        if not(ctr.img_no in img_no):
            test_ctrs_endo.remove(ctr)

    img_test_endo, mask_test_endo = export_all_contours(test_ctrs_endo,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
                                                                                                                             
    img_test_epi, mask_test_epi = export_all_contours(test_ctrs_epi,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size) 
                                                        
    mask_test_myo = np.array(mask_test_epi) - np.array(mask_test_endo)
    
    # replace with your own test random indices
    # you can use random function here!
    random_indices_test = [1, 122, 46, 0, 45, 28, 94, 52, 60, 85, 50, 123, 38, 6, 59, 75, 53, 20, 118, 63, 7, 35, 121, 30, 101, 51, 40, 105, 79, 33, 84, 29, 76, 90, 120, 109, 112, 73, 70, 5, 18, 102, 39, 83, 47, 80, 36, 21, 68, 92]
            
    mask_val_endo = np.array([mask_test_endo[i] for i in random_indices_test])

    # validation data
    img_val = np.array([img_test_endo[i] for i in random_indices_test])
    ctr_test = [test_ctrs_endo [i] for i in random_indices_test]
    #print(ctr_test)

    mask_val_epi = np.array([mask_test_epi[i] for i in random_indices_test])

    mask_val_myo = np.array([mask_test_myo[i] for i in random_indices_test])
    
    # supervised train data
    # replace with your own train indices. (the train indices should be exactly the same as the train data that used for only supervised(line 53)
    # don't use random functions here, it will change the training data!!! use random functions to extract the training data before training the supervised only model and copy the exact same indices here.
    random_indices_train = [27, 102, 30, 66, 90, 74, 11, 118, 85, 98, 53, 61, 81, 101, 42, 84, 86, 79, 77, 62]

    # the corresponding myocardium contours regarding the random train indices
    mask_train_endo = np.array([mask_endo[i] for i in random_indices_train])
    img_train = np.array([img_endo[i] for i in random_indices_train])
    ctr_train = [endo_ctrs [i] for i in random_indices_train]
    
    # the corresponding epicardium contours regarding the random train indices
    mask_train_epi = np.array([mask_epi[i] for i in random_indices_train])

    mask_train_myo = np.array([mask_myo[i] for i in random_indices_train])
    
    save_dir = 'redundant_co_training'
    print('\nProcessing val '+contour_type+' contours...')
    val_ctrs = map_all_contours(VAL_CONTOUR_PATH, contour_type, shuffle=False)
    
    val_ctrs=list(val_ctrs)
    
    # generate the pseudo-labels of the unlabeled data using our redundant co-training approach!!!
    imgs_val , pseudo_val = create_pseudo_labels(val_ctrs, VAL_IMG_PATH)

    print('\nProcessing online '+contour_type+' contours...')
    online_ctrs = map_all_contours(ONLINE_CONTOUR_PATH, contour_type, shuffle=False)
    online_ctrs= list(online_ctrs)
    
    # generate the pseudo-labels of the unlabeled data using our redundant co-training approach!!! (second part)
    imgs_on , pseudo_on = create_pseudo_labels(online_ctrs, ONLINE_IMG_PATH)
    print('\nAll done.')

    
    print('\nBuilding Train dataset ...')

    train_len = (np.array(img_train).shape)[0]
    val_len = (np.array(imgs_val).shape)[0]
    on_len = (np.array(imgs_on).shape)[0]

    # concatenate all the labeled and unlabeled data for SSL re-training
    img_concat = np.concatenate((imgs_on, imgs_val, img_train), axis = 0)

    # use weight = 0.1 for the unlabeled data and 1 for the labeled data
    SSL_weights = np.concatenate ((0.1* np.ones(val_len + on_len), np.ones(train_len)), axis =0)
    print(img_concat.shape)
    
    # Use this line if the primary task is endocardium segmentation 
    mask_train = mask_train_endo
    # otherwise use one of these lines for epicardium or myocardium segmentation
    #mask_train = mask_train_myo
    #mask_train = mask_train_epi

    mask_concat = np.concatenate ((pseudo_on ,pseudo_val, mask_train), axis = 0)   
    
    
    #print(mask_concat.shape)
    
    # validation part
    print('\nBuilding Dev dataset ...')

    # use this line if the primary task is endocardium segmentation
    mask_dev = mask_val_endo
    img_dev = img_val
    
    # otherwise use these lines for epicardium or myocardium segmentation
    
    #mask_dev = mask_val_myo    
    #mask_dev = mask_val_epi    
                                        
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    
    # if you are using the FCN model use this line
    model = fcn_model(input_shape, num_classes, weights=None)
    #print(model.summary())
    
    # if you are useing the U-Net change line 316 with this line 
    #model = unet(input_size = input_shape, pretrained_weights=None)   
    
    #augmentations
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

    # epchs and batches
    epochs = 40
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_concat, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_concat, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    max_iter = ((img_concat.shape[0]) / mini_batch_size) * epochs
    curr_iter = 0
    
    # learning rate
    base_lr = 1*K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_concat)/mini_batch_size)):
            img, mask = next(train_generator)
            # re-train
            res = model.train_on_batch(img, mask, sample_weight = SSL_weights[iteration])
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        # validation
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['dataset_name', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('red_CT'):
            os.makedirs('red_CT')
        save_path = os.path.join('red_CT', save_file)
        #print(save_path)
        model.save_weights(save_path)

