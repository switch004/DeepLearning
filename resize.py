# coding: utf-8
import os
import sys
import numpy as np
import cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob


def resize(input_dir, output_dir, h, w):
    dir_list = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for dir in dir_list:
        image_files = glob.glob(os.path.join(input_dir, dir, "*.jpg"))
        result = os.path.join(output_dir, dir)
        if not os.path.exists(result):
            os.mkdir(result)
        for cnt, image_file in enumerate(image_files):
            print(str(cnt+1) + '/' + str(len(image_files)))
            img = Image.open(image_file)
            resized = img.resize((w, h))
            resized.save(os.path.join(result, str(cnt+1).zfill(3) + '.jpg'))



