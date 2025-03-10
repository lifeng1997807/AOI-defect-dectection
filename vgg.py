# %%
from shutil import copyfile, move
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
import ast
import datetime as dt
import os
import time
from math import trunc
import tensorflow as tf
from keras import backend as K
import cv2
import json
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
from collections import defaultdict
from pathlib import Path
import keras
import warnings
from keras import models, layers, optimizers, losses, metrics, regularizers
from tensorflow.python.keras.layers.core import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import  preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
#check version of tensorflow and test whether uses GPU
print(tf.__version__)
print(tf.test.is_gpu_available())

# 設定 GPU 記憶體增長模式
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ 記憶體增長模式已啟用")
    except RuntimeError as e:
        print(e)

# %%

#import csv file
training_df = pd.read_csv(r"C:\Users\benker\Downloads\aoi\train.csv")


src = "train_images_trans/"
dst = "training_data/"

os.mkdir(dst)
os.mkdir(dst+"0")
os.mkdir(dst+"1")
os.mkdir(dst+"2")
os.mkdir(dst+"3")
os.mkdir(dst+"4")
os.mkdir(dst+"5")

with tqdm(total=len(list(training_df.iterrows()))) as pbar:
    for idx, row in training_df.iterrows():
        pbar.update(0)
        if row["Label"] == 0:
            copyfile(src+row["ID"], dst+"0/"+row["ID"])
        elif row["Label"] == 1:
            copyfile(src+row["ID"], dst+"1/"+row["ID"])
        elif row["Label"] == 2:
            copyfile(src+row["ID"], dst+"2/"+row["ID"])
        elif row["Label"] == 3:
            copyfile(src+row["ID"], dst+"3/"+row["ID"])
        elif row["Label"] == 4:
            copyfile(src+row["ID"], dst+"4/"+row["ID"])
        elif row["Label"] == 5:
            copyfile(src+row["ID"], dst+"5/"+row["ID"])

src = "training_data/"
dst = "validation_data/"

os.mkdir(dst)
os.mkdir(dst+"0")
os.mkdir(dst+"1")
os.mkdir(dst+"2")
os.mkdir(dst+"3")
os.mkdir(dst+"4")
os.mkdir(dst+"5")

validation_df = training_df.sample(n=int(len(training_df)/10))

with tqdm(total=len(list(validation_df.iterrows()))) as pbar:
    for idx, row in validation_df.iterrows():
        pbar.update(0)
        if row["Label"] == 0:
            move(src+"0/"+row["ID"], dst+"0/"+row["ID"])
        elif row["Label"] == 1:
            move(src+"1/"+row["ID"], dst+"1/"+row["ID"])
        elif row["Label"] == 2:
            move(src+"2/"+row["ID"], dst+"2/"+row["ID"])
        elif row["Label"] == 3:
            move(src+"3/"+row["ID"], dst+"3/"+row["ID"])
        elif row["Label"] == 4:
            move(src+"4/"+row["ID"], dst+"4/"+row["ID"])
        elif row["Label"] == 5:
            move(src+"5/"+row["ID"], dst+"5/"+row["ID"])

# %%
#using data augmentation
batch_size = 24
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

train_data_dir = "training_data"
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    shuffle=True,
    target_size=(100, 100),
    batch_size=24,
    class_mode='categorical')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_data_dir = "validation_data"
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(100, 100),
    batch_size=16,
    class_mode='categorical')

input_shape = (100,100,3)
num_classes = 6

#plot images
sample_training_images, _ = next(train_generator)
def plotImages(images_arr): 
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])

#building the training network. you can change the network you want
model_vgg = VGG16(include_top = False,input_shape=(100,100,3),weights = 'imagenet')
model = BatchNormalization()(model_vgg.output)
model = Flatten(name = 'flatten')(model_vgg.output)
model = Dense(6,activation='softmax')(model)

#建立損失函數

def Focal_Loss(alpha=0.25, gamma=2):
    def focal_loss(y_true, y_pred):
        y_pred = K.epsilon() + y_pred  # 防止 log(0) 錯誤
        ce = -y_true * K.log(y_pred)  # 計算交叉熵
        weight = K.pow(1 - y_pred, gamma) * y_true  # 計算權重
        fl = ce * weight * alpha  # 計算損失
        return K.mean(fl, axis=-1)  
    return focal_loss


print(type(model))
#model_vgg = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
model_vgg = Model(model_vgg.input,model,name = 'vgg16')
#model_vgg.compile(optimizer =  'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])  

alpha = [2528/595,2528/443,2528/95,2528/348,2528/220,2528/575]
model_vgg.compile(optimizer =  'adam', loss=[Focal_Loss(alpha=alpha, gamma=2)],metrics = ['accuracy'])  
epochs = 60
lr_scheduler = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1)

history = model_vgg.fit(train_generator,
          validation_data=validation_generator,
          epochs=epochs,
          verbose=1,
          shuffle=False,
          callbacks=[lr_scheduler])

# plot model loss & save
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('vggnet_try2_Loss_summary_graph.png')
plt.show()


# plot model accuracy & save
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('vggnet_try2_Accuracy_summary_graph.png')
plt.show()

# #### Testing on test_data
# 設定資料夾路徑
test_data_dir = r"C:\Users\benker\Downloads\aoi\test_images_trans"

# 取得所有圖片的檔名
image_files = [f for f in os.listdir(test_data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
df = pd.DataFrame({'filename': image_files})
test_datagen=ImageDataGenerator(rescale=1./255.)

#flow_from_dataframe
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=test_data_dir,  # 這裡要加 directory，讓它知道圖片在哪
    x_col='filename',  
    target_size=(100, 100),  
    batch_size=1,  
    class_mode=None,  # 測試集不需要類別
    shuffle=False
)


filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model_vgg.predict(test_generator, steps=len(df), verbose=1)
#sample probability result of one images 
print(predict[0])
label = np.where(predict[0]==max(predict[0]))
label_map = train_generator.class_indices
print(label_map)


#create new file submission for see the result
csv_file = open("vggnet.csv","w")
csv_file.write("ID,Label\n")
for filename, prediction in zip(filenames,predict):
    name = filename.split("/")
    name = name[0]
    label = np.where(prediction==max(prediction))
    label = label[0][0]
    csv_file.write(str(name)+","+str(label)+"\n")
csv_file.close()



import numpy as np

# 確保 filenames 定義
filenames = test_generator.filenames

# 預測結果 (你已經有了 predict 和 filenames)
csv_file = open("vggnet.csv", "w")
csv_file.write("ID,Label\n")

for filename, prediction in zip(filenames, predict):
    # 使用 os.path.basename() 提取檔案名稱，不包含路徑
    name = os.path.basename(filename)
    
    # 使用 np.argmax() 找到預測的最大值索引
    label = np.argmax(prediction)  # 取得最大值的索引，對應到類別
    
    # 寫入 CSV 文件
    csv_file.write(f"{name},{label}\n")

csv_file.close()

# %%
#confution matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from tensorflow.keras.utils import to_categorical
#initial train
train_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    shuffle=False,
    target_size=(100, 100),
    batch_size=1,
    class_mode='categorical')
# 取得模型預測
predictions = model_vgg.predict(train_generator, steps=len(train_generator), verbose=1)

# 取得真實標籤 (一維類別索引)
y_true = train_generator.classes  # shape: (256,)

# 將預測機率轉換為類別索引
y_pred = np.argmax(predictions, axis=1)  # shape: (256,)

# 計算混淆矩陣
cm = skm.confusion_matrix(y_true=y_true, y_pred=y_pred)

plt.figure(figsize=(10, 6))
labels = list(train_generator.class_indices.keys())  # 獲取類別名稱
sns.heatmap(
    cm, xticklabels=labels, yticklabels=labels,
    annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix', fontsize=15)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# %%













