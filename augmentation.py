# -*- coding:utf-8 -*
import os
import math
import random
import glob
import numpy as np
from scipy import misc
from PIL import Image
import cv2
import random
from scipy.misc import imresize
import math
from scipy.ndimage.interpolation import rotate
from shuffle import imwrite
from shuffle import imread

#左右反転
def horizontal_flip(image):
    hflip_img = cv2.flip(image, 1)
    return hflip_img

#コントラスト調整
def contrast(image):
    #ルックアップテーブルの作成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8')
    LUT_LC = np.arange(256, dtype = 'uint8')

    #ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255*(i - min_table)/ diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    #ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    high_cont_img = cv2.LUT(image, LUT_HC)
    low_cont_img = cv2.LUT(image, LUT_LC)
    if random.randint(0,1) == 0:
        return high_cont_img
    else:
        return low_cont_img

#ガンマ変換(輝度変更)
def brightness(image):
    #ルックアップテーブルの作成
    gamma1 = 0.75
    gamma2 = 1.5
    LUT_G1 = np.arange(256, dtype = 'uint8')
    LUT_G2 = np.arange(256, dtype = 'uint8')

    for i in range(256):
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    high_gamma_img = cv2.LUT(image, LUT_G1)
    low_gamma_img = cv2.LUT(image, LUT_G2)
    if random.randint(0,1) == 0:
        return high_gamma_img
    else:
        return low_gamma_img

#cutout
def cutout(image_origin, mask_size):
    # 最後に使うfill()は元の画像を書き換えるので、コピーしておく
    image = np.copy(image_origin)
    mask_value = image.mean()

    h, w, _ = image.shape
    # マスクをかける場所のtop, leftをランダムに決める
    # はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    # はみ出した場合の処理
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    # マスク部分の画素値を平均値で埋める
    image[top:bottom, left:right, :].fill(mask_value)
    return image

def random_crop(image, crop_size):
    h, w, _ = image.shape

    #画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    #top, leftから画像のサイズを足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    #決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[top: bottom, left:right, :]
    return image

def scale_augmentation(image, crop_size):
    scale_range=(crop_size[1]+5,crop_size[1]*2)
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (math.floor(scale_size*1.15), scale_size))
    image = random_crop(image, crop_size)
    return image

def random_rotation(image, angle_range=(0,360)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    scale = 1.0
    oy, ox = int(image.shape[0]/2), int(image.shape[1]/2)

    R = cv2.getRotationMatrix2D((ox, oy), angle, scale)
    dst = cv2.warpAffine(image, R, dsize=(w, h), flags=cv2.INTER_CUBIC)
    #image = rotate(image, angle)
    #image = imresize(image, (h,w))
    return dst

#ガウス分布に基づくノイズ
'''
def gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss_img = image + gauss
    return gauss_img
'''

'''
#ごま塩ノイズ(インパルスノイズ)
def impulse_noise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    sp_img = image.copy()

    #塩モード
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    sp_img[coords[:-1]] = (255, 255, 255)

    #ごまモード
    num_pepper = np.ceil(amount * image.size * (1. -s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    sp_img[coords[:-1]] = (0,0,0)
    return sp_img
'''

'''
#上下反転
def vertical_flip(image):
    vflip_img = cv2.flip(image, 0)
    return vflip_img
'''

'''
#輝度の変更
def random_brightness(image, max_delta=63, seed=None):
    img = np.array(image)
    delta = np.random.uniform(-max_delta, max_delta)
    image = Image.fromarray(np.uint8(img + delta))
    return image
'''

#データ拡張
#data_numに指定した値になるまで「左右反転」「輝度の変更」「コントラストの変更」する
def data_augmentation(image_files, crop_size):
    image_list = []
    file_num = len(image_files)

    for image_file in image_files:
        image_list.append(cv2.imread(image_file, 1))#ここcv2.imreadにしたら高速化可能

    flipped_image = horizontal_flip(image_list[0])
    image_list.append(flipped_image)

    cutout_image = cutout(image_list[0], image_list[0].shape[0] // 2)
    image_list.append(cutout_image)

    rotated_image = random_rotation(image_list[0])
    image_list.append(rotated_image)

    scaled_image = scale_augmentation(image_list[0], crop_size)
    image_list.append(scaled_image)

    bri_img = brightness(image_list[0])
    image_list.append(bri_img)

    contrast_image = contrast(image_list[0])
    image_list.append(contrast_image)

    '''
    if file_num >= data_num:
        return image_list
    '''

    #data_numを使う場合
    '''
    count = len(image_list)
    #print(count)

    #flip left right
    for i, image in enumerate(image_list):
        flipped_image = horizontal_flip(image)
        image_list.append(flipped_image)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break

    
    count = len(image_list)
    #print(count)
    #cutout
    for i, image in enumerate(image_list):
        cutout_image = cutout(image, image.shape[0] // 2)
        image_list.append(cutout_image)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break
    

    count = len(image_list)
    #print(count)
    #random rotation
    for i, image in enumerate(image_list):
        rotated_image = random_rotation(image)
        image_list.append(rotated_image)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break

    count = len(image_list)
    #print(count)
    #scale augmentation
    for i, image in enumerate(image_list):
        scaled_image = scale_augmentation(image, crop_size)
        image_list.append(scaled_image)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break

    count = len(image_list)
    #print(count)
    #brightness
    for i, image in enumerate(image_list):
        bri_img = brightness(image)
        image_list.append(bri_img)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break

    
    count = len(image_list)
    #print(count)
    #contrast
    for i, image in enumerate(image_list):
        contrast_image = contrast(image)
        image_list.append(contrast_image)
        if len(image_list) == data_num:
            return image_list
        elif (i+1) == count:
            break
    

    count = len(image_list)
    #print(count)
    for i, image in enumerate(image_list):
        flipped_image = horizontal_flip(image)
        image_list.append(flipped_image)
        if len(image_list) == data_num:
            return image_list
    '''

    
    # #data_numを使わない場合
    # count = len(image_list)
    # #flip left right
    # for i, image in enumerate(image_list):
    #     flipped_image = horizontal_flip(image)
    #     image_list.append(flipped_image)
    #     if (i+1) == count:
    #         break

    # count = len(image_list)
    # #print(count)
    # #cutout
    # for i, image in enumerate(image_list):
    #     cutout_image = cutout(image, image.shape[0] // 2)
    #     image_list.append(cutout_image)
    #     if (i+1) == count:
    #         break

    # count = len(image_list)
    # #random rotation
    # for i, image in enumerate(image_list):
    #     rotated_image = random_rotation(image)
    #     image_list.append(rotated_image)
    #     if (i+1) == count:
    #         break

    # '''
    # count = len(image_list)
    # #print(count)
    # #scale augmentation
    # for i, image in enumerate(image_list):
    #     scaled_image = scale_augmentation(image, crop_size)
    #     image_list.append(scaled_image)
    #     if (i+1) == count:
    #         break
    # '''

    # count = len(image_list)
    # #print(count)
    # #brightness
    # for i, image in enumerate(image_list):
    #     bri_img = brightness(image)
    #     image_list.append(bri_img)
    #     if (i+1) == count:
    #         break

    # count = len(image_list)
    # #print(count)
    # #contrast
    # for i, image in enumerate(image_list):
    #     contrast_image = contrast(image)
    #     image_list.append(contrast_image)
    #     if (i+1) == count:
    #         break
    
    return image_list

def augmentation(input_dir, output_dir, crop_size):
    dir_list = os.listdir(input_dir)
    #N = 0 #教師用画像枚数
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for dir in dir_list:
        image_files = glob.glob(os.path.join(input_dir, dir, "*.jpg"))
        if len(image_files) == 0:
            continue
        print(dir)

        image_list = data_augmentation(image_files, crop_size)
        result = os.path.join(output_dir, dir)
        if not os.path.exists(result):
            os.mkdir(result)

        for i, image in enumerate(image_list):
            print(str(i+1) + '/' + str(len(image_list)))
            cv2.imwrite(os.path.join(result, str(i+1).zfill(3) + '.jpg'), image)

        #N = N + len(image_list)

    #return N, len(dir_list)
