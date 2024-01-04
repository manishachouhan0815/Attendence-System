#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import time
cam = ['http://192.168.29.211:8080/video',0]
url = 'http://192.168.29.211:8080/video'
k = 0

images = []
#getting 10 images
while(k<10):
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    images.append(frame)
    k+=1


# In[3]:


# while(True):
#     cv2.imshow('frame',images[3])
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()


# In[4]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
model = tf.keras.models.load_model('D:/OneDrive/Desktop/ANN_model')


# In[19]:


import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class Image_Processing:
    def __init__(self):
        """
        :param image:images path
        """
        #creating object of mtcnn model
        self.faces = []               #list to store faces
        self.names = []               #list to store names
        self.features = []            #list to store facenet features
        self.facenet = FaceNet()      #loading the facenet model from Keras library
        self.mtcnn_model = MTCNN()    #loading the mtcnn model

    def face_extractor(self,images):
        """
        :param images: path of images
        :return: array
        """
        print("Extracting Faces......")
        #intializing faces list
        
#         images = os.listdir(images)     #listing the images paths
        print(len(images),"Images found")
        for i in range(0,len(images)):
#             imagepath = os.path.join(os.getcwd()+'\images' , image)
#             img = cv2.imread(os.path.join('D:/OneDrive/Desktop/images',image))                   #reading the image
#             print(image)
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            result = self.mtcnn_model.detect_faces(img)     #extracting face from image
            dimensions = result[0]["box"]                   #getting dimensions from mtcnn result
            x1, y1, width, height = dimensions
            x2, y2 = x1 + width, y1 + height
            img = img[y1:y2, x1:x2]
            self.faces.append(img)          #appending the face extracted to a list
#             self.names.append(image[:-4])   #appending it's name to a list
        print("Faces Extracted Successfully")
#         return self.faces,self.names
        
        """
        :param faces:array containing the faces
        :param names:array containing names

        In this method, we are finding face embeddings using Keras FACENET Model
        """
        print("Extracting features from faces......")
        
        for face in faces:
            face = cv2.resize(face, (160, 160))     
            face = np.reshape(face, (1, 160, 160, 3))
            embeddings = self.facenet.embeddings(face)    
            self.features.append([embeddings])             
#         arr_update = [i.reshape(-1).tolist() for i in np.array(self.features,dtype=object)[:, 0]]
#         df = pd.DataFrame(arr_update)
#         df['Names'] = np.array(self.names,dtype=object)
        print("Features Extracted Successfully")
        return self.faces,self.names,self.features


# In[20]:


imageprocessor = Image_Processing()
faces,names,features = imageprocessor.face_extractor(images)
# print(features)


# In[21]:


prediction = []
for i in range(0,len(images)):
    output = model.predict(features[i])
    x = np.argmax(output)
    arr = np.array(output)
    if(arr[0,x]<0.95):
        prediction.append('unknown')
    else:
        prediction.append(x)
    


# In[23]:


def most_frequent(List):
    return max(set(List), key = List.count)
  
print(most_frequent(prediction))


# In[ ]:




