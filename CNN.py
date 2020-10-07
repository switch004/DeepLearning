# -*- coding: utf-8 -*
import os
import numpy as np
import six
from PIL import Image
import glob
import time
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers
import xlwt
from chainer.backends import cuda
from chainer import Function
from chainer import Link, ChainList
from chainer import cuda
from shuffle import shuffle
from augmentation import augmentation
from resize import resize
from excel import change_excel
from edit import edit_excel

def image2Train(pathsAndLabels, channels=3):

    allData = []
    count_train_label = np.zeros(G)

    #データの追加(画像ファイル，ラベル)
    for pathsAndLabel in pathsAndLabels:
        path = pathsAndLabel[0]
        label = pathsAndLabel[1]
        imagelist = glob.glob(path + '*')
        count_train_label[int(label)] = len(imagelist)
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    #チャンネルが1の時は，画像ファイルとラベルデータに追加
    if channels == 1:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathsAndLabel[0])
            imgData = np.asarray([np.float32(img)/255.0])#画像の正規化
            imageData.append(imgData)
            labelData.append(np.int32(pathsAndLabel[1]))

        threshold = np.int32(len(imageData)/8*7)#データの何割を教師データとテストデータにするか
        train = (imageData[0:threshold], labelData[0:threshold])
        test = (imageData[threshold:], labeData[threshold:])

    #RGBの時の処理
    else:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathAndLabel[0])
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

        dataset = {}
        dataset['train_img'] = np.array(imageData[0:N]).transpose(0,1,2,3)
        dataset['train_label'] = np.array(labelData[0:N])


    return (dataset['train_img'], dataset['train_label']), count_train_label


def image2Test(pathsAndLabels, channels=3):

    allData = []
    count_test_label = np.zeros(G)

    #データの追加(画像ファイル，ラベル)
    for pathsAndLabel in pathsAndLabels:
        path = pathsAndLabel[0]
        label = pathsAndLabel[1]
        imagelist = glob.glob(path + '*')
        count_test_label[int(label)] = len(imagelist)
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    #チャンネルが1の時は，画像ファイルとラベルデータに追加
    if channels == 1:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathsAndLabel[0])
            imgData = np.asarray([np.float32(img)/255.0])#画像の正規化
            imageData.append(imgData)
            labelData.append(np.int32(pathsAndLabel[1]))

        threshold = np.int32(len(imageData)/8*7)#データの何割を教師データとテストデータにするか
        train = (imageData[0:threshold], labelData[0:threshold])
        test = (imageData[threshold:], labeData[threshold:])

    else:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            img = Image.open(pathAndLabel[0])
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

        dataset = {}
        dataset['test_img'] = np.array(imageData[0:N_test]).transpose(0,1,2,3)
        dataset['test_label'] = np.array(labelData[0:N_test])


    return (dataset['test_img'], dataset['test_label']), count_test_label

#畳み込み層2層、プーリング層2層、全結合層3層のCNN
class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 40, 5), # 入力:3 , 出力:40。両方記述。filter 5
            conv2 = L.Convolution2D(40, 50, 5), # 入力:40 , 出力:50。両方記述。filter 5
            l1 = L.Linear(None, 500),
            l2 = L.Linear(500, 500),
            l3 = L.Linear(500, G, initialW=np.zeros((G, 500), dtype=np.float32))
        )
    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

#AlexNet
class AlexNet(Chain):
    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=2)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, G)

    def forward(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

if __name__ == '__main__':
    book = xlwt.Workbook()
    k = 5 #試行回数
    n_epoch = 30 #更新回数
    batch_size = 32 #バッチサイズ
    cnt_trial = 1#試行回数ごとの写真フォルダ分けに使用
    h = 75#入力画像の高さ
    w = 65#入力画像の幅


    input_dir = 'C:\\Users\\user\\Desktop\\input'#入力画像フォルダ
    #input_testing_dir = 'C:\\Users\\user\\Desktop\\input_test'#入力評価用画像フォルダ
    main_dir = 'C:\\Users\\user\\Desktop\\result'#結果出力フォルダ

    #結果出力フォルダがなければ作成
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    #誤判別時の画像出力フォルダ作成
    miss_judge_train_dir = 'C:\\Users\\user\\Desktop\\miss_judge_train'
    miss_judge_test_dir = 'C:\\Users\\user\\Desktop\\miss_judge_test'

    #誤判別時画像出力フォルダがなければ，作成
    if not os.path.exists(miss_judge_train_dir):
        os.mkdir(miss_judge_train_dir)
    if not os.path.exists(miss_judge_test_dir):
        os.mkdir(miss_judge_test_dir)
    for j in range(1, k+1):
        os.mkdir(os.path.join(miss_judge_train_dir, str(j)+'試行目'))
        os.mkdir(os.path.join(miss_judge_test_dir, str(j)+'試行目'))
        for l in range(1, n_epoch+1):
            os.mkdir(os.path.join(os.path.join(miss_judge_train_dir, str(j)+'試行目'), str(l)))
            os.mkdir(os.path.join(os.path.join(miss_judge_test_dir, str(j)+'試行目'), str(l)))

    #ここから学習と判別
    for i in range(1, k+1):
        print(str(i) + '試行目')
        current_dir = os.path.join(main_dir, str(i)+'試行目')
        #current_dirがなければ，作成
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        #shuffleにより評価用画像枚数を返す．
        #N_testはテスト枚数,flagが0ならinputdirからランダムに1枚抜き出し，評価用画像とし，残りを教師用画像とする．
        #flagが1なら教師用フォルダから教師用に，評価用フォルダから評価用にわける．
        N_test = shuffle(input_dir, current_dir, flag = 0)
        train_dir = os.path.join(current_dir, 'training')
        test_dir = os.path.join(current_dir, 'testing')
        #data augmentationの実行
        N, G = augmentation(os.path.join(train_dir, 'image1'), os.path.join(train_dir, 'image2'), 100, (h,w))
        #教師用、評価用ともに画像サイズの変更
        resize(os.path.join(train_dir, 'image2'), os.path.join(train_dir, 'image3'), h, w)
        resize(os.path.join(test_dir, 'image1'), os.path.join(test_dir, 'image2'), h, w)

        #時間計測
        start_time = time.clock()
        pathsAndLabels = []
        paths = []
        for j in range(0, G):
            pathsAndLabels.append(np.asarray([os.path.join(os.path.join(train_dir, 'image3'), 'stage'+str(j)+'\\'), j]))

        for j in range(0, G):
            paths.append(np.asarray([os.path.join(os.path.join(test_dir, 'image2'), 'stage'+str(j)+'\\'), j]))


        (x_train, t_train), correct_train_label = image2Train(pathsAndLabels)
        (x_test, t_test), correct_test_label = image2Test(paths)


        model = CNN()#モデルの選択

        #GPUの設定
        cuda.get_device(0).use()
        model.to_gpu(0)

        optimizer = optimizers.Adam() # 最適化関数
        optimizer.setup(model)

        #ラベルごとのトータル誤判別数
        miss_train_total = np.zeros(G)
        miss_test_total = np.zeros(G)

        #ラベルごとのトータル正解数
        correct_train_total = np.zeros(G)
        correct_test_total = np.zeros(G)

        #何を何と判別したか2次元配列
        miss_train_judge = np.zeros([G,G])
        miss_test_judge = np.zeros([G,G])

        cnt_train = 0#画像を保存するように使用
        cnt_test = 0#画像を保存するときの数

        print(str(i)+'試行目')
        #エクセル結果出力用
        sheet = book.add_sheet(str(i)+'試行目')
        sheet.write(1, 0, '教師用')
        sheet.write(2, 0, '評価用')
        sheet.write(6, 0, '教師用')
        sheet.write(6, G+3, '評価用')


        for l in range(0, G):
            sheet.write(7, l+1, 'stage'+str(l))

        for l in range(0, G):
            sheet.write(7, l+G+4, 'stage'+str(l))

        sheet.write(n_epoch+15, 0, '誤判別(教師用)')
        sheet.write(n_epoch+15, 3, '誤判別先')
        for l in range(0,G):
            sheet.write(n_epoch+16, l+1, l)
            sheet.write(n_epoch+l+17, 0, l)

        sheet.write(n_epoch+15, G+3, '誤判別(評価用)')
        for l in range(0, G):
            sheet.write(n_epoch+16, l+G+4, l)
            sheet.write(n_epoch+l+17, G+3, l)

        for epoch in range(1, n_epoch+1):
            #訓練
            train_loss = 0
            train_accuracy = 0
            perm = np.random.permutation(N)
            miss_train_copy = miss_train_total.copy()
            miss_test_copy = miss_test_total.copy()
            correct_train_copy = correct_train_total.copy()
            correct_test_copy = correct_test_total.copy()

            for i in range(0, N, batch_size):
                x = Variable(cuda.to_gpu(x_train[perm[i:i+batch_size]]))#教師画像データ
                t = Variable(cuda.to_gpu(t_train[perm[i:i+batch_size]]))#教師ラベル
                y = model.forward(x)
                loss = F.softmax_cross_entropy(y, t)  # 活性化関数と損失関数
                model.zerograds()
                loss.backward()

                #誤判別を特定するプログラム
                acc, data, data2, index, pred, corre = F.accuracy(y, t)
                acc.to_cpu()
                data.to_cpu()
                data2.to_cpu()
                index.to_cpu()
                pred.to_cpu()
                corre.to_cpu()
                t.to_cpu()
                x.to_cpu()
                if epoch == n_epoch:
                    for i in range(len(data.data)):
                        miss_train_total[data.data[i]] += 1
                        miss_train_judge[data.data[i]][data2.data[i]] += 1

                for i in range(len(corre.data)):
                    correct_train_total[corre.data[i]] += 1
                
                #誤判別した画像を出力するプログラム
                for inde in index.data:
                    e = np.array(x.data[inde])
                    r = np.array(e[0])#赤色
                    g = np.array(e[1])#緑色
                    b = np.array(e[2])#青色
                    img = np.dstack((np.dstack((r,g)),b))
                    plt.imsave('C:\\Users\\user\\Desktop\\miss_judge_train\\' +str(cnt_trial)+'試行目\\'+str(epoch)+'\\'+str(cnt_train)+ '_epoch_' +str(epoch)+'_'+str(t.data[inde])+'to'+str(pred.data[inde])+'.jpg',img)
                    cnt_train += 1

                optimizer.update()
                train_loss += loss.data*len(t.data)
                train_accuracy += acc.data*len(t.data)

            #評価
            test_loss = 0
            test_accuracy = 0
            for i in range(0, N_test, batch_size):
                x = Variable(cuda.to_gpu(x_test[i:i+batch_size]))#評価画像データ
                t = Variable(cuda.to_gpu(t_test[i:i+batch_size]))#評価ラベル
                y = model.forward(x)
                model.zerograds()
                loss = F.softmax_cross_entropy(y, t)

                #誤判別を特定するプログラム
                acc, data, data2, index, pred, corre  = F.accuracy(y, t)
                acc.to_cpu()
                data.to_cpu()
                data2.to_cpu()
                index.to_cpu()
                pred.to_cpu()
                corre.to_cpu()
                t.to_cpu()
                x.to_cpu()
                if epoch == n_epoch:
                    for i in range(len(data.data)):
                        miss_test_total[data.data[i]] += 1
                        miss_test_judge[data.data[i]][data2.data[i]] += 1

                for i in range(len(corre.data)):
                    correct_test_total[corre.data[i]] += 1

                
                 #誤判別した画像を出力するプログラム
                for inde in index.data:
                    e = np.array(x.data[inde])
                    r = np.array(e[0])#赤色
                    g = np.array(e[1])#緑色
                    b = np.array(e[2])#青色
                    img = np.dstack((np.dstack((r,g)),b))
                    plt.imsave('C:\\Users\\user\\Desktop\\miss_judge_test\\' +str(cnt_trial)+'試行目\\'+str(epoch)+'\\'+str(cnt_test)+ '_epoch_' +str(epoch)+'_'+str(t.data[inde])+'to'+str(pred.data[inde])+'.jpg',img)
                    cnt_test += 1
                
                test_loss += loss.data*len(t.data)
                test_accuracy += acc.data*len(t.data)

            #epochごとの誤判別数
            miss_train_each = miss_train_total - miss_train_copy
            miss_test_each = miss_test_total - miss_test_copy

            #epochごとの正解数
            correct_train_each = correct_train_total - correct_train_copy
            correct_test_each = correct_test_total - correct_test_copy

            #ターミナル上に出力
            print("epoch:{}  train acc:{:.04f}  test acc:{:.04f}".format(epoch, train_accuracy/N, test_accuracy/N_test))
            print('miss_train_total : ' + str(miss_train_total))
            print('miss_train_each : ' + str(miss_train_each))
            print('miss_test_total : ' + str(miss_test_total))
            print('miss_test_each :' + str(miss_test_each))
            print('correct_train_each :' + str(correct_train_each))
            print('correct_test_each :' + str(correct_test_each))
            print()

            sheet.write(0, epoch, epoch)#単純にエポック数(現時点)
            sheet.write(1, epoch, round(train_accuracy/N, 4))#教師用画像の判別率
            sheet.write(2, epoch, round(test_accuracy/N_test, 4))#評価用画像の判別率

            sheet.write((epoch + 7), 0, epoch)
            sheet.write((epoch + 7), G+3, epoch)
            for i in range(len(correct_train_each)):
                sheet.write((epoch+7), (i + 1), round(correct_train_each[i]/correct_train_label[i], 4))
                sheet.write((epoch + 7), (i + G+4), round(correct_test_each[i]/correct_test_label[i], 4))

        #最終epochの誤判定時における、何が何を予測したのかを表示
        for i in range(0, G):
            for j in range(0,G):
                if j==i:
                    sheet.write((n_epoch+17+i), (i+1), correct_train_each[i])
                    sheet.write((n_epoch+17+i), (i+G+4), correct_test_each[i])
                else:
                    sheet.write((n_epoch+17+i), (j+1), miss_train_judge[i][j])
                    sheet.write((n_epoch+17+i), (j+G+4), miss_test_judge[i][j])

        cnt_trial+=1
        end_time = time.clock()
        sheet.write(0,0,end_time-start_time)
        print(end_time - start_time)


    book.save(os.path.join(main_dir, 'result.xls'))

    change_excel(os.path.join(main_dir, 'result.xls'))
    os.remove(os.path.join(main_dir, 'result.xls'))
    
    #edit_excel(os.path.join(main_dir, 'result.xlsx'), G)
    
