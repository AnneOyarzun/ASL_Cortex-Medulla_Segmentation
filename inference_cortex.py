import os
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import tensorflow as tf
import time
import pandas as pd
import skimage.color
import cv2 as cv
from mrcnn import utils
from preprocessing import specific_intensity_window_1
from mrcnn.config import Config
import preprocessing

class KidneyConfig(Config):
    NAME = 'kidney'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.85
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    BACKBONE='resnet50'
    IMAGE_SHAPE = [256, 256, 3]
    STEPS_PER_EPOCH = 150
    DETECTION_MAX_INSTANCES = 2
    LEARNING_RATE = 0.0002
    USE_MINI_MASK = False
    RPN_ANCHOR_RATIOS = [0.5,1,2]

class InferenceConfig(KidneyConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.85
        NUM_CLASSES = 1 + 2

def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    dice = 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def evaluate_tk(model, data_path, studies, predictions_path):
    t_prediction = 0
    slice_results_df = pd.DataFrame(columns=['Study', 'ImageNumber', 'SliceNumber', 'DiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'SliceNumber', 'DiceScoreAccumulated'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = data_path + 'images/' + nstudies
        masks_path = data_path + 'masks/' + nstudies

        for nslice in range(0, 3):
            pred_sum = None
            gt_sum = None
            for nimgs in range(1, 52):
                images = sitk.ReadImage(images_path + f'{nimgs}.nii')
                gt_masks = sitk.ReadImage(masks_path + f'{nimgs}.mha')
                gt_masks = preprocessing.fill_holes(gt_masks)

                image = images[:, :, nslice]
                gt_mask = gt_masks[:, :, nslice]
                image = preprocess_img(image, rescale=True, int_window=True)
                gt_mask = preprocess_img(gt_mask)

                # Run detection
                t = time.time()
                r = model.detect([image], verbose=0)[0]
                t_prediction += (time.time() - t)

                # Determine the largest mask as kidney
                try:
                    volume1 = r["masks"][:, :, 0].astype(bool)
                    dice1 = calculate_dice(gt_mask[:, :, 0], volume1)
                    try:
                        volume2 = r["masks"][:, :, 1].astype(bool)
                        dice2 = calculate_dice(gt_mask[:, :, 0], volume2)
                        if dice1 >= dice2:
                            pred_mask = volume1
                            dice = dice1
                        else:
                            pred_mask = volume2
                            dice = dice2
                    except:
                        pred_mask = volume1
                        dice = dice1
                except:
                    print('No detection is made')
                    pred_mask = np.zeros_like(image[:, :, 0])  # Si no se detecta nada, usar una máscara vacía
                    dice = 0

                if pred_sum is None:
                    pred_sum = pred_mask.astype(np.int16)
                    gt_sum = gt_mask[:, :, 0].astype(np.int16)
                else:
                    pred_sum += pred_mask.astype(np.int16)
                    gt_sum += gt_mask[:, :, 0].astype(np.int16)

                # Save slice-level results
                slice_results_df = slice_results_df.append({
                    'Study': filename.split('_')[0],
                    'ImageNumber': nimgs,
                    'SliceNumber': nslice + 1,
                    'DiceScore': dice
                }, ignore_index=True)

                mask_filename = f'{filename}Slice_{nslice + 1}_nIm_{nimgs}.nii'
                os.makedirs(predictions_path + 'seg/', exist_ok=True)
                mask_filepath = os.path.join(predictions_path + 'seg/', mask_filename)
                pred_mask = pred_mask.astype('uint8')
                resized_pred = resize_image(sitk.GetImageFromArray(pred_mask), (96,96), is_mask=True)
                sitk.WriteImage(resized_pred, mask_filepath)

            # Filtrar las máscaras acumuladas
            total_pred_sum = pred_sum > 25
            total_gt_sum = gt_sum > 25 

            # Calcular Dice Score para las máscaras filtradas
            dice_pred_sum = calculate_dice(total_gt_sum, total_pred_sum)
            print(f'Study: {filename.split("_")[0]}, Slice: {nslice + 1}, Dice Score Accumulated: {dice_pred_sum}')

            # Save accumulated Dice score to DataFrame
            dice_accumulated_df = dice_accumulated_df.append({
                'Study': filename.split('_')[0],
                'SliceNumber': nslice + 1,
                'DiceScoreAccumulated': dice_pred_sum
            }, ignore_index=True)

    return slice_results_df

def evaluate_shk(model, images_path, masks_path, studies, predictions_path):
    '''
    Las imágenes están divididas en Right y Left. 306 máscaras. 
    '''
    t_prediction = 0
    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    
    for nstudies in studies:
        print('Evaluating study: ', nstudies)        
        
        for nimgs in range(2,52): 
            
            for nslice in range(1,4): 
                image = sitk.ReadImage(images_path + f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}.nii')
                image_rescaled = sitk.RescaleIntensity(image, 0, 255)
                image_resized = resize_image(image_rescaled, (3, 96, 96), is_mask=False)

                gt_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_{nimgs}_Slice_{nslice}.nii')
                gt_mask = preprocess_img(gt_mask)

                # Run detection
                t = time.time()
                r = model.detect([image_resized], verbose=0)[0]
                t_prediction += (time.time() - t)

                # For Right detection
                try:
                    volume1 = r["masks"][:,:,0].astype(bool)
                    dice1 = calculate_dice(gt_mask[:,:,0], volume1)
                    try:
                        volume2 = r["masks"][:,:,1].astype(bool)
                        dice2 = calculate_dice(gt_mask[:,:,0], volume2)
                        if dice1 >= dice2:
                            pred_mask = volume1
                            dice = dice1
                        elif dice2 > dice1:
                            pred_mask = volume2
                            dice = dice2
                    except: 
                        pred_mask = volume1
                        dice = dice1
                except:
                    print('No detection is made')
                    dice = 0


                # Store the results in the DataFrame
                filename = f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}'
                results_df = results_df.append({'Filename': filename, 
                                                'DiceScore': dice, 
                                                }, ignore_index=True)
                
                # pred_mask.astype('uint8')
                # mask_rgb = skimage.color.gray2rgb(pred_mask)
                # mask_filepath = os.path.join(predictions_path + 'seg/', filename + '.nii')
                # os.makedirs(predictions_path + 'seg/', exist_ok=True)
                # sitk.WriteImage(sitk.GetImageFromArray(mask_rgb), mask_filepath)

    return results_df

def evaluate_stk(model, images_path, masks_path, studies, predictions_path):
    '''
    Las imágenes están divididas en Right y Left. 306 máscaras. 
    '''
    t_prediction = 0
    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreRightAccumulated', 'DiceScoreLeftAccumulated'])

    for nstudies in studies:
        print('Evaluating study: ', nstudies)  

        pred_sum_right = None
        pred_sum_left = None
        gt_sum_right = None
        gt_sum_left = None
        
        for nimgs in range(2, 52): 
            for nslice in range(1, 4): 
                image = sitk.ReadImage(images_path + f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}.nii')
                image_resized = resize_image(image, (3, 96, 96), is_mask = False)
                image_blurred = preprocessing.blur_half_img(sitk.GetArrayFromImage(image_resized))
                right_img = image_blurred[0, :, :, 0]
                right_img = preprocess_img(sitk.GetImageFromArray(right_img), rescale=True, int_window=False)
                left_img = image_blurred[0, :, :, 1]
                left_img = preprocess_img(sitk.GetImageFromArray(left_img), rescale=True, int_window=False)

                gt_right_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_{nimgs}_Right_Slice_{nslice}.nii')
                gt_right_mask = resize_image(gt_right_mask, (256,256, 1), is_mask=True)
                gt_right_mask = sitk.GetArrayFromImage(gt_right_mask[:,:,0])
                gt_left_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_{nimgs}_Left_Slice_{nslice}.nii')
                gt_left_mask = resize_image(gt_left_mask, (256,256, 1), is_mask=True)
                gt_left_mask = sitk.GetArrayFromImage(gt_left_mask[:,:,0])


                # Run detection
                t = time.time()
                r_right = model.detect([right_img], verbose=0)[0]
                r_left = model.detect([left_img], verbose=0)[0]
                t_prediction += (time.time() - t)

                # For Right detection
                try:
                    right_volume1 = r_right["masks"][:,:,0].astype(bool)
                    right_dice1 = calculate_dice(gt_right_mask, right_volume1)
                    try:
                        right_volume2 = r_right["masks"][:,:,1].astype(bool)
                        right_dice2 = calculate_dice(gt_right_mask, right_volume2)
                        if right_dice1 >= right_dice2:
                            right_pred_mask = right_volume1
                            right_dice = right_dice1
                        else:
                            right_pred_mask = right_volume2
                            right_dice = right_dice2
                    except: 
                        right_pred_mask = right_volume1
                        right_dice = right_dice1
                except:
                    print('No detection is made')
                    right_pred_mask = np.zeros_like(right_img[:,:,0])
                    right_dice = 0

                # For Left detection
                try:
                    left_volume1 = r_left["masks"][:,:,0].astype(bool)
                    left_dice1 = calculate_dice(gt_left_mask, left_volume1)
                    try:
                        left_volume2 = r_left["masks"][:,:,1].astype(bool)
                        left_dice2 = calculate_dice(gt_left_mask, left_volume2)
                        if left_dice1 >= left_dice2:
                            left_pred_mask = left_volume1
                            left_dice = left_dice1
                        else:
                            left_pred_mask = left_volume2
                            left_dice = left_dice2
                    except: 
                        left_pred_mask = left_volume1
                        left_dice = left_dice1
                except:
                    print('No detection is made')
                    left_pred_mask = np.zeros_like(left_img[:,:,0])
                    left_dice = 0

                # Acumular las máscaras predichas y de verdad terreno
                if pred_sum_right is None:
                    pred_sum_right = right_pred_mask.astype(np.int16)
                    gt_sum_right = gt_right_mask.astype(np.int16)
                else:
                    pred_sum_right += right_pred_mask.astype(np.int16)
                    gt_sum_right += gt_right_mask.astype(np.int16)

                if pred_sum_left is None:
                    pred_sum_left = left_pred_mask.astype(np.int16)
                    gt_sum_left = gt_left_mask.astype(np.int16)
                else:
                    pred_sum_left += left_pred_mask.astype(np.int16)
                    gt_sum_left += gt_left_mask.astype(np.int16)

                # Store the results in the DataFrame
                filename = f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}'
                results_df = results_df.append({'Filename': filename, 
                                                'RightDiceScore': right_dice, 
                                                'LeftDiceScore': left_dice}, ignore_index=True)
                
                total_mask = right_pred_mask.astype('uint8') + left_pred_mask.astype('uint8')
                total_mask_resized = resize_image(sitk.GetImageFromArray(total_mask), (96, 96), is_mask = True)
                mask_filepath = os.path.join(predictions_path + 'seg/', filename + '.nii')
                os.makedirs(predictions_path + 'seg/', exist_ok=True)
                sitk.WriteImage(total_mask_resized, mask_filepath)

        # Filtrar las máscaras acumuladas
        num_slices = (50 - 2) * 3  # Total number of slices evaluated
        total_pred_sum_right = pred_sum_right > np.round(num_slices / 2)
        total_gt_sum_right = gt_sum_right > np.round(num_slices / 2)

        total_pred_sum_left = pred_sum_left > np.round(num_slices / 2)
        total_gt_sum_left = gt_sum_left > np.round(num_slices / 2)

        # Calcular Dice Score acumulado para las máscaras filtradas
        dice_pred_sum_right = calculate_dice(total_gt_sum_right, total_pred_sum_right)
        dice_pred_sum_left = calculate_dice(total_gt_sum_left, total_pred_sum_left)
        
        print(f'Study: {nstudies}, Right Dice Score Accumulated: {dice_pred_sum_right}')
        print(f'Study: {nstudies}, Left Dice Score Accumulated: {dice_pred_sum_left}')

        # Save accumulated Dice score to DataFrame
        dice_accumulated_df = dice_accumulated_df.append({
            'Study': nstudies,
            'DiceScoreRightAccumulated': dice_pred_sum_right,
            'DiceScoreLeftAccumulated': dice_pred_sum_left
        }, ignore_index=True)

    return results_df, dice_accumulated_df

def evaluate_GVox_hk(model, img_path, studies, predictions_path):

    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreRightAccumulated', 'DiceScoreLeftAccumulated'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = img_path + nstudies + 'with_M0/images/result.nii'
        masks_path = img_path + nstudies + 'with_M0/masks/Cortex/result.nii'

        images = sitk.ReadImage(images_path)
        gt_masks = sitk.ReadImage(masks_path) # toda la serie

        pred_sum_right = None
        pred_sum_left = None
        gt_sum_right = None
        gt_sum_left = None

        for nslice in range(1, images.GetSize()[2]): 
            image = images[:, :, nslice]
            image = preprocess_img(image, rescale=True, int_window=False)
            image_resized = resize_image(sitk.GetImageFromArray(image), (3, 96, 96), is_mask=False)
            
            image_blurred = preprocessing.blur_half_img(sitk.GetArrayFromImage(image_resized))
            right_img = image_blurred[0, :, :, 0]
            right_img = preprocess_img(sitk.GetImageFromArray(right_img), rescale=True, int_window=False)
            left_img = image_blurred[0, :, :, 1]
            left_img = preprocess_img(sitk.GetImageFromArray(left_img), rescale=True, int_window=False)

            gt_mask = gt_masks[:, :, nslice]
            gt_right_mask, gt_left_mask = preprocessing.label_right_left(sitk.GetArrayFromImage(gt_mask))
            gt_right_mask = resize_image(sitk.GetImageFromArray(gt_right_mask), (256,256))
            gt_left_mask = resize_image(sitk.GetImageFromArray(gt_left_mask), (256,256))
            gt_right_mask = sitk.GetArrayFromImage(gt_right_mask)
            gt_left_mask = sitk.GetArrayFromImage(gt_left_mask)

            # Run detection
            t = time.time()
            r_right = model.detect([right_img], verbose=0)[0]
            r_left = model.detect([left_img], verbose=0)[0]

            # For Right detection
            try:
                right_volume1 = r_right["masks"][:, :, 0].astype(bool)
                right_dice1 = calculate_dice(gt_right_mask, right_volume1)
                try:
                    right_volume2 = r_right["masks"][:, :, 1].astype(bool)
                    right_dice2 = calculate_dice(gt_right_mask, right_volume2)
                    if right_dice1 >= right_dice2:
                        right_pred_mask = right_volume1
                        right_dice = right_dice1
                    else:
                        right_pred_mask = right_volume2
                        right_dice = right_dice2
                except: 
                    right_pred_mask = right_volume1
                    right_dice = right_dice1
            except:
                print('No detection is made')
                right_pred_mask = np.zeros_like(image[:, :, 0])
                right_dice = 0

            # For Left detection
            try:
                left_volume1 = r_left["masks"][:, :, 0].astype(bool)
                left_dice1 = calculate_dice(gt_left_mask, left_volume1)
                try:
                    left_volume2 = r_left["masks"][:, :, 1].astype(bool)
                    left_dice2 = calculate_dice(gt_left_mask, left_volume2)
                    if left_dice1 >= left_dice2:
                        left_pred_mask = left_volume1
                        left_dice = left_dice1
                    else:
                        left_pred_mask = left_volume2
                        left_dice = left_dice2
                except: 
                    left_pred_mask = left_volume1
                    left_dice = left_dice1
            except:
                print('No detection is made')
                left_pred_mask = np.zeros_like(image[:, :, 0])
                left_dice = 0

            if pred_sum_right is None:
                pred_sum_right = right_pred_mask.astype(np.int16)
                gt_sum_right = gt_right_mask.astype(np.int16)
            else:
                pred_sum_right += right_pred_mask.astype(np.int16)
                gt_sum_right += gt_right_mask.astype(np.int16)

            if pred_sum_left is None:
                pred_sum_left = left_pred_mask.astype(np.int16)
                gt_sum_left = gt_left_mask.astype(np.int16)
            else:
                pred_sum_left += left_pred_mask.astype(np.int16)
                gt_sum_left += gt_left_mask.astype(np.int16)

            # Store the results in the DataFrame
            print('Right Dice: ', right_dice)
            print('Left Dice: ', left_dice)

            total_filename = filename + f'nIm_{nslice}'
            results_df = results_df.append({'Filename': total_filename, 
                                            'RightDiceScore': right_dice, 
                                            'LeftDiceScore': left_dice}, ignore_index=True)
            
            total_mask = right_pred_mask.astype('uint8') + left_pred_mask.astype('uint8')
            # total_mask_rgb = skimage.color.gray2rgb(total_mask)
            total_mask_resized = resize_image(sitk.GetImageFromArray(total_mask), (96,96), is_mask = True)
            mask_filepath = os.path.join(predictions_path + 'seg/', total_filename + '.nii')
            os.makedirs(predictions_path + 'seg/', exist_ok=True)
            sitk.WriteImage(total_mask_resized, mask_filepath)

        # Filtrar las máscaras acumuladas
        total_pred_sum_right = pred_sum_right > np.round((images.GetSize()[2]/2)) # no todos tienen 50 slices
        total_gt_sum_right = gt_sum_right > np.round((images.GetSize()[2]/2))

        total_pred_sum_left = pred_sum_left > np.round((images.GetSize()[2]/2))
        total_gt_sum_left = gt_sum_left > np.round((images.GetSize()[2]/2))

        # Save one slice masks
        total_pred_filename = filename + f'nIm_{nslice}_OneSlicePred'
        total_gt_filename = filename + f'nIm_{nslice}_OneSliceGT'
        total_pred_sum_right_resized = resize_image(sitk.GetImageFromArray(total_pred_sum_right.astype('uint8')), (96,96), is_mask=True)
        total_pred_sum_left_resized = resize_image(sitk.GetImageFromArray(total_pred_sum_left.astype('uint8')), (96,96), is_mask=True)
        total_gt_sum_right_resized = resize_image(sitk.GetImageFromArray(total_gt_sum_right.astype('uint8')), (96,96), is_mask=True)
        total_gt_sum_left_resized = resize_image(sitk.GetImageFromArray(total_gt_sum_left.astype('uint8')), (96,96), is_mask=True)

        sitk.WriteImage(total_pred_sum_right_resized, predictions_path + 'seg/' + total_pred_filename + '_Right.nii')
        sitk.WriteImage(total_pred_sum_left_resized, predictions_path + 'seg/'+ total_pred_filename + '_Left.nii')
        sitk.WriteImage(total_gt_sum_right_resized, predictions_path + 'seg/'+ total_gt_filename + '_Right.nii')
        sitk.WriteImage(total_gt_sum_left_resized, predictions_path + 'seg/'+ total_gt_filename + '_Left.nii')

        # Calcular Dice Score acumulado para las máscaras filtradas
        dice_pred_sum_right = calculate_dice(total_gt_sum_right, total_pred_sum_right)
        dice_pred_sum_left = calculate_dice(total_gt_sum_left, total_pred_sum_left)
        
        print(f'Study: {filename.split("_")[0]}, Right Dice Score Accumulated: {dice_pred_sum_right}')
        print(f'Study: {filename.split("_")[0]}, Left Dice Score Accumulated: {dice_pred_sum_left}')

        # Save accumulated Dice score to DataFrame
        dice_accumulated_df = dice_accumulated_df.append({
            'Study': filename.split('_')[0],
            'DiceScoreRightAccumulated': dice_pred_sum_right,
            'DiceScoreLeftAccumulated': dice_pred_sum_left
        }, ignore_index=True)


    return results_df, dice_accumulated_df

def evaluate_GVox_tk(model, img_path, studies, predictions_path):

    results_df = pd.DataFrame(columns=['Filename', 'DiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreAccumulated'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = img_path + nstudies + 'with_M0/images/result.nii'
        masks_path = img_path + nstudies + 'with_M0/masks/Cortex/result.nii'

        images = sitk.ReadImage(images_path)
        gt_masks = sitk.ReadImage(masks_path) # toda la serie
        

        pred_sum = None
        gt_sum = None

        for nslice in range(1, images.GetSize()[2]): 
            image = images[:, :, nslice]
            gt_mask = gt_masks[:,:,nslice]
            image = preprocess_img(image, rescale=True, int_window=False)
            gt_mask = resize_image(gt_mask, (256, 256), is_mask=True)
            gt_mask = sitk.GetArrayFromImage(gt_mask)

            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]

            # For Right detection
            try:
                volume1 = r["masks"][:, :, 0].astype(bool)
                dice1 = calculate_dice(gt_mask, volume1)
                try:
                    volume2 = r["masks"][:, :, 1].astype(bool)
                    dice2 = calculate_dice(gt_mask, volume2)
                    if dice1 >= dice2:
                        pred_mask = volume1
                        dice = dice1
                    else:
                        pred_mask = volume2
                        dice = dice2
                except: 
                    pred_mask = volume1
                    dice = dice1
            except:
                print('No detection is made')
                pred_mask = np.zeros_like(image[:, :, 0])
                dice = 0


            if pred_sum is None:
                pred_sum = pred_mask.astype(np.int16)
                gt_sum = gt_mask.astype(np.int16)
            else:
                pred_sum += pred_mask.astype(np.int16)
                gt_sum += gt_mask.astype(np.int16)

            total_filename = filename + f'nIm_{nslice}'
            results_df = results_df.append({'Filename': total_filename, 
                                            'DiceScore': dice}, ignore_index=True)
            
            total_mask_resized = resize_image(sitk.GetImageFromArray(pred_mask.astype('uint8')), (96,96), is_mask=True)
            mask_filepath = os.path.join(predictions_path + 'seg/', total_filename + '.nii')
            os.makedirs(predictions_path + 'seg/', exist_ok=True)
            sitk.WriteImage(total_mask_resized, mask_filepath)

        # Filtrar las máscaras acumuladas
        total_pred_sum = pred_sum > np.round((images.GetSize()[2]/2)) # no todos tienen 50 slices
        pred_sum = total_pred_sum.astype('uint8')
        pred_sum = resize_image(sitk.GetImageFromArray(pred_sum), (96,96), is_mask=True)
        total_gt_sum = gt_sum > np.round((images.GetSize()[2]/2))
        gt_sum = total_gt_sum.astype('uint8')
        gt_sum = resize_image(sitk.GetImageFromArray(gt_sum), (96,96), is_mask=True)
        total_filename_predsum = filename + f'nIm_{nslice}_OneSlicePred'
        total_filename_gtsum = filename + f'nIm_{nslice}_OneSliceGT'
        sitk.WriteImage(pred_sum, predictions_path + 'seg/' + total_filename_predsum + '.nii')
        sitk.WriteImage(gt_sum, predictions_path + 'seg/' + total_filename_gtsum + '.nii')

        # Calcular Dice Score acumulado para las máscaras filtradas
        dice_pred_sum = calculate_dice(total_gt_sum, total_pred_sum)
        
        print(f'Study: {filename.split("_")[0]}, Dice Score Accumulated: {dice_pred_sum}')

        # Save accumulated Dice score to DataFrame
        dice_accumulated_df = dice_accumulated_df.append({
            'Study': filename.split('_')[0],
            'DiceScoreAccumulated': dice_pred_sum
        }, ignore_index=True)


    return results_df, dice_accumulated_df

def resize_image(image, new_size, is_mask=False):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [(original_spacing[i] * original_size[i] / new_size[i]) for i in range(len(new_size))]
    
    # Choose the interpolation method based on whether the image is a mask
    if is_mask:
        interpolation_method = sitk.sitkNearestNeighbor
    else:
        interpolation_method = sitk.sitkLinear
    
    resampled_image = sitk.Resample(image, new_size, sitk.Transform(), interpolation_method,
                                    image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                                    image.GetPixelID())
    return resampled_image

def preprocess_img(image, rescale = None, int_window = None): 
    image_resized = resize_image(image, (256, 256), is_mask=False)
    image_array = sitk.GetArrayFromImage(image_resized)
    # image_array.astype('float32')
    image_rgb = skimage.color.gray2rgb(image_array)
    if rescale:
        image_rescaled1 = sitk.RescaleIntensity(sitk.GetImageFromArray(image_rgb), 0, 255)
        if int_window:
            image_rescaled2 = preprocessing.specific_intensity_window_1(sitk.GetArrayFromImage(image_rescaled1), window_percent=0.15)
            return image_rescaled2
        else:
            return sitk.GetArrayFromImage(image_rescaled1)
    if int_window and not rescale:
        image_rescaled = preprocessing.specific_intensity_window_1((image_rgb), window_percent=0.15)
        return image_rescaled

    else:
        return image_rgb

def crop_and_mask_image(image, mask):
    # Utilizar el filtro para calcular las estadísticas de forma en la imagen etiquetada
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)
    
    # Inicializar la imagen resultante como una imagen en negro
    result_image = sitk.GetArrayFromImage(image).copy()
    result_image[:] = 0  # Hacer todo negro

    # Convertir la imagen a formato numpy
    image_np = sitk.GetArrayFromImage(image)
    
    # Procesar cada etiqueta (1 y 2)
    for label in label_shape_filter.GetLabels():
        bounding_box = label_shape_filter.GetBoundingBox(label)
        
        # Extraer la información de la caja delimitadora
        x_min, y_min, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x_max, y_max = x_min + width, y_min + height
        
        # Recortar la imagen y la máscara para la etiqueta actual
        cropped_region = image_np[y_min:y_max, x_min:x_max]
        
        # Aplicar la máscara dentro de la caja delimitadora
        result_image[y_min:y_max, x_min:x_max] = cropped_region
    
    # Convertir la imagen procesada de vuelta a SimpleITK
    result_image_sitk = sitk.GetImageFromArray(result_image)
    result_image_sitk.CopyInformation(image)  # Mantener la información del espacio original
    
    return result_image_sitk

def calculate_metrics(gt_mask, pred_mask, beta=2):
    # Ensure the masks are binary (0 and 1)
    gt_mask = gt_mask.astype(np.bool_)
    pred_mask = pred_mask.astype(np.bool_)
    
    # True Positives (TP): Both ground truth and prediction are 1
    TP = np.sum((gt_mask == 1) & (pred_mask == 1))
    
    # False Positives (FP): Prediction is 1 but ground truth is 0
    FP = np.sum((gt_mask == 0) & (pred_mask == 1))
    
    # False Negatives (FN): Ground truth is 1 but prediction is 0
    FN = np.sum((gt_mask == 1) & (pred_mask == 0))
    
    # Precision calculation
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall calculation
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F-measure calculation with beta = 2
    beta_squared = beta ** 2
    f_measure = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (beta_squared * precision + recall) > 0 else 0
    
    # Dice coefficient calculation
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    return precision, recall, f_measure, dice

if __name__ == '__main__': 
    ###################################################
    ########### LOAD MODEL ##############################
    ###################################################

    logs_dir = ''
    model_dir = logs_dir
    weights_path = os.path.join(model_dir, '.h5')

    mode = 'inference'

    config = InferenceConfig()

    model = modellib.MaskRCNN(
        mode=mode,
        config=config,
        model_dir=logs_dir, 
        )

    print('Loading weights...')
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name = True)

    ###################################################################################
    # DATA PATHS
    ###################################################################################
    
    result_path = 'Z:/RM_RENAL/Segmentation/Multiclass_segmentation/Segmentation_Results/MaskRCNN/'
    excel_path = result_path

    ###################################################################################
    # INFERENCE ON TK T1-SERIES (14 images in the series)
    ###################################################################################
    

    # r = model.detect([image], verbose=0)[0]

    # # Detectar corteza
    # try:
    #     cortex_volume = r["masks"][:, :, 0].astype(bool)
    # except:
    #     print('No cortex detection is made')
    #     cortex_volume = np.zeros((96, 96), dtype=np.uint8)

    # # Detectar médula
    # try:
    #     medulla_volume = r["masks"][:, :, 1].astype(bool)
    # except:
    #     print('No medulla detection is made')
    #     medulla_volume = np.zeros((96, 96), dtype=np.uint8)

    # precision_cortex, recall_cortex, f_measure_cortex, dice_cortex = calculate_metrics(gt_cortex, pred_cortex)

                    