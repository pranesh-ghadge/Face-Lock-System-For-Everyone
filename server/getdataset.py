import numpy as np
import cv2
import time
import os
import shutil
import joblib
import json
from util import get_cropped_image_if_2_eyes
from wavelet import w2d
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

__class_name_to_number = {}
__class_number_to_name = {}

timeout = 30
timeout_start = time.time()
cap = cv2.VideoCapture(0)

image_folder_path = "./model/dataset/full_images/person/"
path_to_data = "./dataset/full_images/"
path_to_cr_data = "./dataset/cropped/"


if os.path.exists(image_folder_path):
    shutil.rmtree(image_folder_path)
os.mkdir(image_folder_path)

count_of_image = 1

while time.time() < timeout_start + timeout :
    ret, frame = cap.read()

    image = np.zeros(frame.shape, np.uint8) 
    image = frame

    cv2.imshow('frame',image)

    if cv2.waitKey(1) == ord('q'):
        break

    imagepath = image_folder_path + "training_image" + str(count_of_image) + ".png"

    cv2.imwrite(imagepath, image)
    count_of_image +=1


if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

individual_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    individual_name = img_dir.split('/')[-1]
    
    individual_file_names_dict[individual_name] = []
    
    cropped_individual_folder = path_to_cr_data + individual_name
    os.mkdir(cropped_individual_folder)
    
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:          
            cropped_file_name = "image_cropped" + str(count) + ".png"
            cropped_file_path = cropped_individual_folder + '/' + cropped_file_name 
            
            cv2.imwrite(cropped_file_path, roi_color)
            individual_file_names_dict[individual_name].append(cropped_file_path)
            count += 1    

with open("./server/artifacts/class_dictionary.json", "r") as f:
    __class_name_to_number = json.load(f)
    __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}


X, y = [], []

for cropped_dir in os.scandir(path_to_cr_data):
    individual_name = cropped_dir.path.split('/')[-1]
    
    for training_image in os.scandir(cropped_dir):
        img = cv2.imread(training_image.path)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(__class_number_to_name[individual_name]) 

X = np.array(X).reshape(len(X),4096).astype(float)


X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, random_state=42)

xgbst = XGBClassifier(max_depth=3, min_child_weight=3, scale_pos_weight=3)
pipe_tt_xgb = Pipeline([('scaler', StandardScaler()), ('xgbclassifier', xgbst)])
pipe_tt_xgb.fit(X_train_split, y_train_split)

# Save the model as a pickle in a file 
joblib.dump(pipe_tt_xgb, 'saved_model.pkl') 

src_path = "./server/saved_model.pkl"
dst_path = "./server/artifacts/saved_model.pkl"

shutil.copy(src_path, dst_path)