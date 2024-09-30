from joblib import load
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

classes=['No Glaucoma','Glaucoma']

model=load("./final_model.joblib")

def predict_and_plot(path):
    img=cv.imread(path)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    resized=cv.resize(gray, (290,290))
    x_var=np.array(resized)
    x_var=x_var.reshape(-1,290*290)/255
    predicted=model.predict(x_var)
    plt.imshow(img)
    plt.title(classes[predicted[8]])
    plt.show()

path=r'./train//class1//BEH-105.png'
predict_and_plot(path)
path=r'./test//class0//BEH-203.png'
predict_and_plot(path)