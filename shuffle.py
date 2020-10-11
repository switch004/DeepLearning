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
import xlwt
import math

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def shuffle(input_dir, output_dir, input_testing_dir=None, flag=0):
    book = xlwt.Workbook()
    sheet = book.add_sheet('画像枚数')
    sheet.write(0, 0, 'ラベル')
    sheet.write(0, 1, '葉齢')
    sheet.write(0, 2, '教師用画像枚数')
    sheet.write(0, 3, '評価用画像枚数')
    sheet.write(0, 4, '合計')
    #ディレクトリのリストを取得
    dir_list = [filename for filename in os.listdir(input_dir) if not filename.startswith('.')]

    dir_list.sort(key=int)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_dir = os.path.join(output_dir, 'training')
    test_dir = os.path.join(output_dir, 'testing')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_image1 = os.path.join(train_dir, 'image1')
    test_image1 = os.path.join(test_dir, 'image1')
    if not os.path.exists(train_image1):
        os.mkdir(train_image1)
    if not os.path.exists(test_image1):
        os.mkdir(test_image1)

    i=0
    cnt_test_image = 0

    #ディレクトリを取得
    if flag == 0:
        dir_list = [filename for filename in os.listdir(input_dir) if not filename.startswith('.')]
        dir_list.sort(key=int)
        print(dir_list)
        for dir in dir_list:
            image_files = glob.glob(os.path.join(input_dir, dir, "*.JPG"))
            print(dir)
            if len(image_files)<2:
                continue

            image_list = []
            train_image_list = []#教師用画像を入れるリスト
            test_image_list = []#評価用画像を入れるリスト


            #ディレクトリ内の画像ファイルをリストに保存
            for image_file in image_files:
                image_list.append(imread(image_file, 1))


            #画像が保存されたリストをシャッフル
            image_list = np.random.permutation(image_list)
            
            #先頭の画像1枚をテスト用に保存/52枚目までを教師用に
            test_image_list = image_list[0:1]
            train_image_list = image_list[1:52]
            '''
            #先頭からfloor_numまでを教師用画像に、それ以降を評価用画像とする
            floor_num = math.floor(len(image_list)*0.8)
            train_image_list = image_list[0:floor_num]
            test_image_list = image_list[floor_num:]
            '''
            #教師用画像用フォルダ(train)と評価用画像用フォルダ(test)を作成
            if i < 10:
                train_output_dir = os.path.join(train_image1, 'stage0'+str(i))
                test_output_dir = os.path.join(test_image1, 'stage0'+str(i))
            else:
                train_output_dir = os.path.join(train_image1, 'stage'+str(i))
                test_output_dir = os.path.join(test_image1, 'stage'+str(i))

            if not os.path.exists(train_output_dir):
                os.mkdir(train_output_dir)
            if not os.path.exists(test_output_dir):
                os.mkdir(test_output_dir)

            #教師用画像を保存
            for j, image in enumerate(train_image_list):
                cv2.imwrite(os.path.join(train_output_dir, str(j+1).zfill(3) + '.JPG'), image)

            #評価用画像を保存
            for j, image in enumerate(test_image_list):
                cv2.imwrite(os.path.join(test_output_dir, str(j+1).zfill(3) + '.JPG'), image)

            #エクセル記入
            sheet.write(i+1, 0, i)
            sheet.write(i+1, 1, int(dir))#葉齢の番号
            sheet.write(i+1, 2, len(train_image_list))
            sheet.write(i+1, 3, len(test_image_list))
            sheet.write(i+1, 4, len(image_list))

            i = i+1
            cnt_test_image = cnt_test_image + len(test_image_list)

        book.save(os.path.join(output_dir, '画像枚数.xls'))
        return cnt_test_image


    if flag == 1:
        #ディレクトリのリストを取得
        dir_list = [filename for filename in os.listdir(input_dir) if not filename.startswith('.')]
        dir_list.sort(key=int)
        print(dir_list)
        testing_dir_list = [filename for filename in os.listdir(input_testing_dir) if not filename.startswith('.')]
        testing_dir_list.sort(key=int)
        print(testing_dir_list)

        yourei_label = []
        
        #教師用画像のほう
        for dir in dir_list:
            yourei_label.append(int(dir))
            image_files = glob.glob(os.path.join(input_dir, dir, "*.jpg"))
            print(dir)
            if len(image_files)<1:
                continue

            image_list = []
            train_image_list = []

            #ディレクトリ内の画像ファイルをリストに保存
            for image_file in image_files:
                image_list.append(imread(image_file, 1))

            image_list = np.random.permutation(image_list)
            # if len(image_list) > 300:
            #     train_image_list = image_list[0:300]
            # else:
            #     train_image_list = image_list[0:]#必要に応じて変更

            if int(dir) < 12:
                train_image_list = image_list[0:300]
            if int(dir) >= 12:
                train_image_list = image_list[0:103]
            # train_image_list = image_list[0:]

            #教師用画像用フォルダ(train)と評価用画像用フォルダ(test)を作成
            if i < 10:
                train_output_dir = os.path.join(train_image1, 'stage0'+str(i))
                if not os.path.exists(train_output_dir):
                    os.mkdir(train_output_dir)
            else:
                train_output_dir = os.path.join(train_image1, 'stage'+str(i))
                if not os.path.exists(train_output_dir):
                    os.mkdir(train_output_dir)

            #教師用画像を保存
            for j, image in enumerate(train_image_list):
                cv2.imwrite(os.path.join(train_output_dir, str(j+1).zfill(3) + '.JPG'), image)

            #エクセル記入
            sheet.write(i+1, 0, i)
            sheet.write(i+1, 1, int(dir))#葉齢の番号
            sheet.write(i+1, 2, len(train_image_list))

            i = i+1

        book.save(os.path.join(output_dir, '画像枚数.xls'))
        
        #評価用画像のほう
        i = 0
        for testing_dir in testing_dir_list:

            testing_image_files = glob.glob(os.path.join(input_testing_dir, testing_dir, "*.jpg"))


            image_list = []
            test_image_list = []
            for testing_image_file in testing_image_files:
                test_image_list.append(imread(testing_image_file, 1))

            #print(test_image_list)
            if yourei_label.index(int(testing_dir)) < 10:
                test_output_dir = os.path.join(test_image1, 'stage0'+str(yourei_label.index(int(testing_dir))))
            else:
                test_output_dir = os.path.join(test_image1, 'stage'+str(yourei_label.index(int(testing_dir))))

            if not os.path.exists(test_output_dir):
                os.mkdir(test_output_dir)

            #評価用画像を保存
            for j, image in enumerate(test_image_list):
                cv2.imwrite(os.path.join(test_output_dir, str(j+1).zfill(3) + '.JPG'), image)

            sheet.write(yourei_label.index(int(testing_dir))+1, 3, len(test_image_list))
            sheet.write(i+1, 4, len(image_list))

            i = i+1
            cnt_test_image = cnt_test_image + len(test_image_list)
        
        book.save(os.path.join(output_dir, '画像枚数.xls'))
        return cnt_test_image