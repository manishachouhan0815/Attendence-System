import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import json
import warnings
warnings.filterwarnings('ignore')
model = tf.keras.models.load_model('ANN_model')


from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
import os
import pandas as pd
class Image_Processing:
    def __init__(self,image):
        """
        :param image:images path
        """
        #creating object of mtcnn model
        self.facenet = FaceNet()      #loading the facenet model from Keras library
        self.mtcnn_model = MTCNN() #loading the mtcnn model 
        self.image = image
        try:
            f = open('sot.json')
            self.sot = json.load(f)
        except Exception as e:
            print(e)            
    def face_validate(self, image):
        try:
            if image is not None:
                detected_faces = self.mtcnn_model.detect_faces(image)  # detect faces if image is read correctly
                if len(detected_faces) == self.sot["no_of_person_per_image"]:  # if number of face detected is 1
                    if image.shape[2] == self.sot["Layers"]:  # if number of channels in image are 3
                        if len(image.shape) == self.sot["Dimension"]:  # if shape of image is n*n*c then
                            return True
                else:
                    print("Image is not valid")
                    return False
            else:
                print("Image not found")
                return False
        except Exception as e:
            print("some error occurred during face validation")
            print(e)

    def face_extractor(self):  #number is the urls in cams and length is the amount of pictures taken from each url/camera
        try:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            validation = self.face_validate(img)      # face validating image
            if validation:                            # we will continue if the image is valid 
                result = self.mtcnn_model.detect_faces(img)    # face detection
                dimensions = result[0]["box"]                   #getting dimensions from mtcnn result
                x1, y1, width, height = dimensions               
                x2, y2 = x1 + width, y1 + height
                img = img[y1:y2, x1:x2]
                face = cv2.resize(img, (160, 160))     
                face = np.reshape(face, (1, 160, 160, 3))
                embeddings = self.facenet.embeddings(face)   #getting embeddings for prediction 
                embeddings = embeddings.reshape(1,512)
                pred = model.predict(embeddings)
                #saving images to the required path
                x = np.argmax(pred)
                arr = np.array(pred)
                if(arr[0,x]<0.95):
                    result = 'unknown'
                else:
                    result = x
                print(result)
            else:
                print()
        except Exception as e:
            print(e)
            
            
image = cv2.imread('photo.jpg')
imageprocessor = Image_Processing(image)
imageprocessor.face_extractor()