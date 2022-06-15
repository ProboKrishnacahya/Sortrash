### Note: Install terlebih dahulu
# 1. pip install keras
# 2. pip install tensorflow
# 3. pip3 install opencv-python 
# 4. Model langsung di load, jadi tidak perlu melakukan training lagi

import tracemalloc
from keras.models import Sequential
import sys
from keras.layers import Conv2D, Activation, Dropout
from keras.models import Model,load_model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense
import tensorflow.python.keras.engine
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import itertools

tf.random.set_seed(1000)

# !mkdir uas_data

import os
THIS_CODE_PATH = 'F:/UC/1.IMT/1.Kelas_Mata_Kuliah/4.Semester_IV/3.Artificial_Intelligence/2.Tugas/W16/ALP_AI_Archotech_GUI/App'

import shutil
test_dir = THIS_CODE_PATH + '/content/uas_data/TEST'

test_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
        test_dir, 
        target_size= (224,224),
        batch_size = 2500,
        color_mode= "rgb",
        class_mode= "categorical",
        seed= 42)

# """### Load Model"""

model = tf.keras.models.load_model(THIS_CODE_PATH + '/content/cnn_model')

# """# Melakukan prediksi dengan test_x"""

test_x, test_y = test_generator.__getitem__(0) #0 merupakan badge 0 dan perbadge isinya ada ~2.450 gambar

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
i=0
k=0
preds = model.predict(test_x)

def predictMain(img):
  tracemalloc.start()
  img = cv2.imread(img) 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  imgtemp = img
  img = cv2.resize(img,(224,224))

  img = np.expand_dims(img, axis=0)
  img = np.asarray(img,np.float32)/255

  predimg = model.predict(img)
  accScore = round(np.max(predimg)/np.sum(predimg)*100, 2)

  labels = {0: "AnorganikNonKertas", 
  1: "AnorganikKertas", 2: "Organik"}

  import os, psutil
  process = psutil.Process(os.getpid())

  print("")

  print("memory",tracemalloc.get_traced_memory())
  
  # stopping the library
  plt.figure(figsize=(16, 16))
  plt.subplot(4, 4, 1)
  resultLabel = "0"
  if labels[np.argmax(predimg)] == "Organik":
    resultLabel = "Organik"
    descriptionLabel = "Sampah yang berasal dari tumbuh-tumbuhan dan mudah mengalami daur ulang.\nContoh: Kayu, Dedaunan, Bangkai dan Kotoran Hewan, dll"
    plt.title(str(accScore) + "% " + "Organik")
  elif labels[np.argmax(predimg)] == "AnorganikNonKertas":
    resultLabel = "Anorganik Berbahan Non-Kertas"
    descriptionLabel = "Sampah non kertas yang terdiri atas unsur yang tidak dapat diproses secara alami.\nContoh: Botol Plastik, Kaleng, Karet, dll"
    plt.title(str(accScore) + "% " + "Anorganik Berbahan Non-Kertas")
  else:
    resultLabel = "Anorganik Berbahan Kertas"
    descriptionLabel = "Sampah kertas yang terdiri atas unsur yang tidak dapat diproses secara alami.\nContoh: Kertas Hout Vrij Schrift (HVS), Karton, Kraft Paper, dll"

  # """# Check Result"""

  #first run in terminal so can show the result before show the image show
  y_pred = [np.argmax(x) for x in preds]
  y_test = [np.argmax(x) for x in test_y]
  from sklearn.metrics import confusion_matrix, accuracy_score
  from sklearn.metrics import f1_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import precision_score
  print("")
  print("================")
  print("Confusion Matrix")
  print("================")
  print("")
  cf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
  print(cf_matrix)
  
  print("")
  print("================")
  print("Evaluation Model")
  print("================")
  print('Precision: ', precision_score(y_test,y_pred, average="macro"))
  print('Recall: ', recall_score(y_test,y_pred, average="macro"))
  print("Accuracy Score: ", accuracy_score(y_test, y_pred))
  print('f1 Score: ', f1_score(y_test, y_pred, average="macro"))

  from sklearn.metrics import classification_report
  precision = str(precision_score(y_test,y_pred, average="macro"))
  recall = str(recall_score(y_test,y_pred, average="macro"))
  accuracy = str(accuracy_score(y_test, y_pred))
  f1 = str(f1_score(y_test, y_pred, average="macro"))
  print("Memory:",tracemalloc.get_traced_memory())
  tracemalloc.stop()
  
  return resultLabel, descriptionLabel, precision, recall, accuracy, f1
  # model.save('/content/model_1')
