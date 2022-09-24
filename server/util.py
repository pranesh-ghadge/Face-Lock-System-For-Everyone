import joblib
import json
import numpy as np
import shutil
from wavelet import w2d
import cv2
import time
import os

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(file_path):
    imgs = get_cropped_image_if_2_eyes(file_path)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append(class_number_to_name(__model.predict(final)[0]))

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cropped_image_if_2_eyes(image_path):
    face_cascade = cv2.CascadeClassifier('./server/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./server/opencv/haarcascades/haarcascade_eye.xml')

    img = cv2.imread(image_path)

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cropped_faces = []
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
        return cropped_faces

if __name__ == '__main__':
    load_saved_artifacts()

    flag_to_indicate_that_image_is_classified = 0
    count_of_image = 0
    resultant_pred = []

    timeout = 10
    timeout_start = time.time()
    cap = cv2.VideoCapture(0)

    image_folder_path = "./model/images_to_classified/"
    
    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)

    os.mkdir(image_folder_path)

    while time.time() < timeout_start + timeout :
        ret, frame = cap.read()

        image = np.zeros(frame.shape, np.uint8) 
        image = frame

        cv2.imshow('frame',image)

        if cv2.waitKey(1) == ord('q'):
            break

        imagepath = image_folder_path + "image" + str(count_of_image) + ".png"

        cv2.imwrite(imagepath, image)
        count_of_image +=1

    count_of_image = 0

    for image in os.scandir(image_folder_path):
        result_of_identification = classify_image(image.path)

        if len(result_of_identification):
            # flag_to_indicate_that_image_is_classified = 1
            resultant_pred[count_of_image] = 1
        else:
            resultant_pred[count_of_image] = 0

        count_of_image +=1 

    count_of_one=0
    count_of_zero=0

    for i in resultant_pred:
        if i==1:
            count_of_one += 1
        else:
            count_of_zero +=1

    if count_of_one > count_of_zero:
        flag_to_indicate_that_image_is_classified = 1

    if flag_to_indicate_that_image_is_classified == 1:
#   redirect to the homepage
        print('hello')
    else:
        print('Get Lost!')