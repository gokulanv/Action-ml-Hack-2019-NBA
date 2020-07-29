#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,accuracy_score


# In[3]:


''' Class for data loader. It takes the training or evaluation data and creates an object of Sequence class 
from Keras. The class takes care of loading batch of images while training the model and discarding them after use.
This class is used to create objects for model with sift features.
'''
class CNN_Loader(Sequence):
    def __init__(self, img_paths, targets, batch_size):
        self.img_paths, self.targets = img_paths, targets
        self.batch_size = batch_size
        #self.all_y = []

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_imgs = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        s_features = []
        for filename in batch_imgs:
            img = cv2.imread("data/"+filename, cv2.IMREAD_COLOR)
            images.append(np.transpose(image.img_to_array(image.load_img("data/"+filename, target_size=(224, 224))),(1,0,2)))
            blur = cv2.blur(img,(50,50))
            d_temp = np.zeros((25600,))
            blur2 = cv2.GaussianBlur(img,(3,3),0)
            absd=cv2.equalizeHist(cv2.cvtColor(cv2.absdiff(blur2,blur),cv2.COLOR_BGR2GRAY))
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency.computeSaliency(absd)
            threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            ss = saliencyMap.copy()
            ss[:40, :] = 0
            ss[:, :20] = 0
            ss[:, -20:] = 0
            sift1 = cv2.xfeatures2d.SIFT_create(200)
            kp1, desc = sift1.detectAndCompute(ss, None)
            f = desc.flatten()
            if desc is not None:
                d_temp[:min(25600,len(f))] = f[:25600]
            s_features.append(d_temp)
        s_features = np.asarray(s_features)
        return [np.array(images), s_features], np.array(batch_y)
   
''' Class for data loader. It takes the training or evaluation data and creates an object of Sequence class 
from Keras. The class takes care of loading batch of images while training the model and discarding them after use.
This class is used to create objects for model without sift features.
'''

class CNN_Loader_with_sift(Sequence):
    def __init__(self, img_paths, targets, batch_size):
        self.img_paths, self.targets = img_paths, targets
        self.batch_size = batch_size
        #self.all_y = []

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_imgs = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        for filename in batch_imgs:
            images.append(np.transpose(image.img_to_array(image.load_img("data/"+filename, target_size=(224, 224))),(1,0,2)))
        return np.array(images), np.array(batch_y)


# In[4]:


'''This function loads the given model'''
def load_model(name):
    json_file = open('./Models/'+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./Models/"+name+".h5")
    return loaded_model

'''This function creates the image data loader object'''
def get_data_gen(csv_path, sift):
    data = pd.read_csv(csv_path, header=None)
    data=data.sample(frac=1)
    x_train, y_train = data.iloc[:,0].values, data.iloc[:,1].astype(int).values
    return CNN_Loader(x_train, y_train, 32) if sift else CNN_Loader_with_sift(x_train, y_train, 32)

'''This function returns the scores calculated between the predictions and truth values'''
def scoring(pred, all_y):
    accuracy = accuracy_score(all_y, pred)
    precision = precision_score(all_y, pred)
    recall = recall_score(all_y, pred)
    return {'accuracy': accuracy, 'precision':precision, 'recall':recall}

'''This function writes the label on the images and saves them in a new directory for creating a video'''
def write_images(img_paths,pred, dest_path, task_name):
    for i in range(len(img_paths)):
        im = cv2.imread("data/"+img_paths[i])
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "False" if np.round(pred[i])==0 else "True"
        cv2.putText(im, text, (20,160), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(dest_path+"/"+img_paths[i].split("/")[-1], im)

'''This function creates a video using the images stored with labels using the write_images function'''
def create_video(path_to_dir, img_paths):
    img_array = []
    files = os.listdir(path_to_dir)
    files = [i for i in files if "jpg" in i] 
    size = (0, 0)
    for filename in files:
        img = cv2.imread(path_to_dir+"/"+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    out = cv2.VideoWriter(path_to_dir+"/"+'project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

'''The master function that takes the task name and whether or not we use sift features and gets the inference 
scores'''
def inference_model(task_name, sift):
    csv_path = "./data/"+task_name+"_training_data1.csv"
    if sift == True:
        model_name = "CNN_model_xception_sift_full_"+task_name
        dest_path = "data/Final_Predictions/"+task_name+"_with_sift"
    else:
        model_name = "CNN_model_xception_full_"+task_name
        dest_path = "data/Final_Predictions/"+task_name
    model = load_model(model_name)
    print('Model loaded...')
    eval_data_gen = get_data_gen(csv_path, sift)
    pred = model.predict_generator(eval_data_gen)
    print('Got predictions...')
    scores = scoring(np.round(pred), eval_data_gen.targets)
    print(scores)
    print(confusion_matrix(np.round(pred),eval_data_gen.targets))
    try:
        os.mkdir(dest_path)
    except:
        pass
    write_images(eval_data_gen.img_paths, pred, dest_path, task_name)
    create_video(dest_path, eval_data_gen.img_paths)
    print('Video created...')
    play_video(dest_path)
    return scores


# In[9]:


inference_model("three_point", True)
