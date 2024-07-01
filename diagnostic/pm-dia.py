
import sys
import matplotlib.pyplot as plt
import numpy as np
 
np.set_printoptions(suppress=True)
from sklearn import datasets, linear_model
 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
import pandas as pd
from keras.models import Model
 
from sklearn.svm import SVC
 
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
 
from sklearn.metrics import roc_curve, auc
 
import pywt
 
scorelist = []
 
import keras
 
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Input
import numpy as np
import math
from keras.layers import concatenate
 
# In[595]:
 
 
from scipy.io import loadmat
 
import scipy.io as scio
 

from scipy.io import loadmat

import scipy.io as scio
import scipy.io as scio
from sklearn import preprocessing
import h5py


import glob
import numpy as np
import os
import nibabel as nib
import torch
import pydicom
import cv2
import glob

import numpy as np
import os
import nibabel as nib
import cv2
from sklearn import preprocessing

x_tt = []
for k in range (1,116):

   xt_path = os.path.join('D:/code/dcm/2023_3_6_dataset/normal_dcm/{}'.format(k))
   x_t = []
   x_t1 = np.zeros((1, 256, 256))
   count = 0
   for root, dirs, files in os.walk(xt_path ):  # 遍历统计
      for each in files:
         count += 1  # 统计文件夹下文件个数
   #print(count)

   xa_path = os.path.join('D:/code/dcm/2023_3_6_dataset/normal_label')
   x_a = os.path.join(xa_path, 'label'+format(k)+'.nii')
   x_a = nib.load(x_a)
   x_a = x_a.get_data().astype(np.float32)
   x_a = torch.from_numpy(x_a)
   x_a = np.array(x_a)
   slice = x_a.shape[2]
   time=count//slice

   for i in range(0, slice):
      x_a22=x_a[:,:,i]
      if cv2.countNonZero(x_a22) != 0:
         s1=i
         for j in range(0, 3):
            number = (slice * (time // 2 - 1) + s1 +1) + slice * j
            x_t_name = '000' + format(number) + '.dcm'
            x_t = os.path.join(xt_path, x_t_name)
            x_t = ''.join(x_t)
            x_t = pydicom.read_file(x_t)
            x_t = np.array(x_t.pixel_array)
            x_t = cv2.resize(x_t, dsize=(256, 256))
            x_t = x_t.tolist()
            x_tt.append(x_t)
x_tt = np.array(x_tt)
print(x_tt.shape)


x_tt_pm= []
for k in range (1,161):

   xt_path = os.path.join('D:/code/dcm/2023_3_6_dataset/dcm/{}'.format(k))
   x_t_pm = []
   x_t1_pm = np.zeros((1, 256, 256))
   count = 0
   for root, dirs, files in os.walk(xt_path ):  # 遍历统计
      for each in files:
         count += 1  # 统计文件夹下文件个数

   xa_path = os.path.join('D:/code/dcm/2023_3_6_dataset/label')
   x_a_pm = os.path.join(xa_path, 'label'+format(k)+'.nii')
   x_a_pm = nib.load(x_a_pm)
   x_a_pm = x_a_pm.get_data().astype(np.float32)
   x_a_pm = torch.from_numpy(x_a_pm)
   x_a_pm = np.array(x_a_pm)
   slice = x_a_pm.shape[2]
   time=count//slice

   for i in range(0, slice):

      x_a22_pm=x_a_pm[:,:,i]
      if cv2.countNonZero(x_a22_pm) != 0:
         s1=i
         for j in range(0, 3):
            number = (slice * (time // 2 - 1) + s1 +1) + slice * j
            x_t_name = '000' + format(number) + '.dcm'
            x_t_pm = os.path.join(xt_path, x_t_name)
            x_t_pm = ''.join(x_t_pm)
            x_t_pm = pydicom.read_file(x_t_pm)
            x_t_pm = np.array(x_t_pm.pixel_array)
            x_t_pm = cv2.resize(x_t_pm, dsize=(256, 256))
            x_t_pm = x_t_pm.tolist()
            x_tt_pm.append(x_t_pm)
x_tt_pm = np.array(x_tt_pm)
print(x_tt_pm.shape)

X = np.concatenate((x_tt,x_tt_pm)).astype(np.float16)

print(X.shape)

X = X.reshape(2586, 256,256,1)


data = pd.read_csv("D:/code/dcm/label.csv")
#data = pd.read_csv("E:/label-200.csv")
Y = data.iloc[:, 1].values
Y = np_utils.to_categorical(Y, num_classes=2)
print(Y.shape)


def random_shuffle(X, Y):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(X)
    np.random.seed(randnum)
    np.random.shuffle(Y)
    return X, Y


X, Y = random_shuffle(X, Y)
'''

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2);
print(x_train.shape)
'''


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()



from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, Dropout
from keras.layers import Conv1D, MaxPool1D
from keras.layers.convolutional import SeparableConv2D
# from keras.layers.normalization import BatchNormalization
#from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.constraints import max_norm

from keras.layers.advanced_activations import ELU
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
import tensorflow as tf

nbatch_size = 128
nEpoches = 100
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def buildlstm():
    import numpy as np

    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode="valid", subsample=(2, 2),
                            input_shape=(256, 256, 1)))  # output=((227-5)/2 + 1 = 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output=((112-2)/2 + 1 = 56

    model.add(Convolution2D(32, 5, 5, border_mode="same"))  # output = 56
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))  # output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output=((56-2)/2 + 1 = 28

    model.add(Convolution2D(64, 3, 3, border_mode="same"))  # output = 28
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))  # output= 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output=((28-2)/2 + 1 = 14

    model.add(Convolution2D(96, 3, 3, border_mode="same"))  # output = 14
   # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode="valid"))  # output = ((14-3)/1) +1 = 12
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output=((12-2)/2 + 1 = 6

    model.add(Convolution2D(192, 3, 3, border_mode="same"))  # output =6
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3, border_mode="valid"))  # output = ((6-3)/1) + 1 = 4
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output=((4-2)/2 + 1 = 2

    model.add(Flatten())

    model.add(Dense(output_dim=4096, input_dim=2 * 2 * 192))
    model.add(Activation('relu'))
    # model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4))  # for sec level

    model.add(Dense(output_dim=4096, input_dim=4096))
    model.add(Activation('relu'))
    # model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4))  # for sec level

    model.add(Dense(output_dim=2, input_dim=4096))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
    pass


def runTrain(model, x_train, x_test, y_train, y_test):
    history = LossHistory()
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nEpoches, validation_data=(x_test, y_test),
              callbacks=[history], shuffle=True)
    history.loss_plot('epoch')
    score = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('evaluate score:', score)
    pass


def test():
    # print(x_train.shape)
    # model = EEGNet()
    model = buildlstm()
    # print(model.get_weights()[0])
    # runTrain(model, x_train, x_test, y_train, y_test)xiugai
    return model

from sklearn.model_selection import train_test_split

seed = 666
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
import time
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------
import keras.backend as K
import scipy.io as sio
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)

from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, roc_auc_score

i = 0
Loss = []
Acc = []
Val_Acc = []
Val_loss = []
sum = 0
acc = []
sco = []
roc = []
pre = []
rec = []

import joblib

acc_temp = 0
from sklearn.preprocessing import label_binarize

TRAIN_START = time.time()
for train_index, test_index in kf.split(x_train):
    K.clear_session()
    i += 1
    x_train_, y_train_ = x_train[train_index], y_train[train_index]
    x_test_, y_test_ = x_train[test_index], y_train[test_index]
    model = test()
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nEpoches,batch_size=nbatch_size, callbacks=[early_stopping])
    history = model.fit(x_train_, y_train_, validation_data=(x_test_, y_test_), epochs=nEpoches, batch_size=nbatch_size)
    pred = model.predict(x_test_)
    y_pred_ = model.predict([x_test_])
    y_pred_ = np.argmax(y_pred_, axis=1)
    y_true_ = np.argmax(y_test_, axis=1)
    # 添加

    print("KFold第%d组：" % (i))
    Loss.append(history.history['loss'])
    Val_loss.append(history.history['val_loss'])
    Acc.append(history.history['accuracy'])
    Val_Acc.append(history.history['val_accuracy'])

    print("Accuracy:")
    accuray = accuracy_score(y_true_, y_pred_)
    print(accuray)
    acc.append(accuray)
    if accuray >= acc_temp:
        joblib.dump(model, './test.pkl')
        acc_temp = accuray
        print('---------------------', '已保存模型：', i, '---------------------')
    print("Precision:")
    prec = precision_score(y_true_, y_pred_, average=None)
    print(prec)
    pre.append(prec)

    print("F1 Score:")
    f1_sc = f1_score(y_true_, y_pred_, average=None)
    print(f1_sc)
    sco.append(f1_sc)

    print("Recall Score:")
    recall = recall_score(y_true_, y_pred_, average=None)
    print(recall)
    rec.append(recall)

    n_class = 2
    y_one_hot_ = label_binarize(y=y_test_, classes=np.arange(n_class))
    y_score_pro_ = model.predict(x_test_)

    auc_ = roc_auc_score(y_one_hot_, y_score_pro_, average='micro')
    print("ROC-AUC:")
    print(auc_)
    roc.append(auc_)
print("VAL_Accuracy:")
print(np.mean(acc), np.std(acc))
print("VAL_Precision::")
print(np.mean(pre), np.std(pre))
print("VAL_Recall Score:")
print(np.mean(rec), np.std(rec))
print("VAL_F1 Score:")
print(np.mean(sco), np.std(sco))
print("VAL_ROC-AUC:")
print(np.mean(roc), np.std(roc))
sio.savemat('./VAL_accloss.mat', {'loss': Loss, 'val_loss': Val_loss, 'acc': Acc, 'val_acc': Val_Acc})
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------
# ----------------------------TRAIN_修改--------------------------------

# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------

print('----------------------------------------------------------------------------------')
model = joblib.load('./test.pkl')

history = model.fit(x_train, y_train, epochs=nEpoches, batch_size=nbatch_size)
TRAIN_END = time.time()
# pred = model.predict(X_test)
y_pred = model.predict([x_test])
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("TEST_Accuracy:")
print(accuracy_score(y_true, y_pred))
print("TEST_Precision::")
print(precision_score(y_true, y_pred, average=None))
print("TEST_Recall Score:")
print(recall_score(y_true, y_pred, average=None))
print("TEST_F1 Score:")
print(f1_score(y_true, y_pred, average=None))
n_class = 2
y_one_hot = label_binarize(y=y_test, classes=np.arange(n_class))
y_score_pro = model.predict(x_test)

auc = roc_auc_score(y_one_hot, y_score_pro, average='micro')
print("TEST_ROC-AUC:")
print(auc)

print('训练时间为：', (TRAIN_END - TRAIN_START))

# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------
# ----------------------------TEST_修改--------------------------------
