import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from matplotlib.image import imread

model = load_model('fmask.h5')

train_path = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\TFLOW_ENV\\my_shit\\Facemask\\data\\train'
test_path = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\TFLOW_ENV\\my_shit\\Facemask\\data\\test'

label_dict = {0: 'Mask', 1: 'No Mask'}
color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}

# image = cv2.imread(train_path + '\\without_mask\\0.jpg')
cam = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# new_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

while True:
 rv,image = cam.read()
 image = cv2.flip(image,1,1)
 faces = haar.detectMultiScale(image)
 for f in faces:
    (x, y, w, h) = [v for v in f]  # Scale the shapesize backup
    # Save just the rectangle faces in SubRecFaces
    print(x, y, w, h)
    face_img = image
    resized = cv2.resize(face_img, (224, 224))
    normalized = resized / 255
    reshaped = np.reshape(normalized, (1, 224, 224, 3))
    result = model.predict(reshaped)
    print(result)

    # choose the index of the label_dict
    label = np.argmax(result, axis=1)[0]
    print(label)
    text = "{}:{}% ".format(label_dict[label], (np.max(result) * 100) % 100)

    f_mask = cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
    f_mask = cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_dict[label], 2)
    f_mask = cv2.resize(image, (350, 480))

 cv2.imshow('mask',image)
 key=cv2.waitKey(10)
 if key == 27:  # The Esc key
     break

# Stop video
cam.release()

# Close all started windows
cv2.destroyAllWindows()
    # cv2.imwrite('mask.png', f_mask)
    # plt.imshow(f_mask)