# coding: utf-8
import os
import sys
import numpy as np
import cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob


def count_image(input_dir):
    N = 0#教師用画像枚数
    G = 0#グループ数
    dir_list = os.listdir(input_dir)
    G = len(dir_list)
    for dir in dir_list:
        image_files = glob.glob(os.path.join(input_dir, dir, "*.jpg"))
        N = N + len(image_files)
    return N, G



