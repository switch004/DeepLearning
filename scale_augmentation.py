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

#データ拡張
def data_augmentation(image_files, crop_size):
    image_list = []
    file_num = len(image_files)

    for image_file in image_files:
        image_list.append(cv2.imread(image_file, 1))

    count = len(image_list)
    #scale augmentation
    for i, image in enumerate(image_list):
        #print(i)
        scaled_image = scale_augmentation(image, crop_size)
        #print(count)
        image_list.append(scaled_image)
        if (i+1) == count:
            break

    return image_list


def scale_data_augmentation(input_dir, output_dir, crop_size):
    dir_list = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for dir in dir_list:
        image_files = glob.glob(os.path.join(input_dir, dir, "*.JPG"))
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
