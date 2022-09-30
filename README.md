
# Face Lock System

Ready to use Face Lock System for everyone which gives accces only to the user and never allows anyone else. Anyone can set the face-lock for themselves and reset it anytime.


## Features

- **Fully Automatic** Image Capturing, Data Cleaning and Identification by Model.
- **100% Accees Denied** to 'person other than the user'.
- Access Denied to person with **Closed Eyes**.
- Very High Specificity, Recall ~< Specificity. 


## Tech Stack

**Programming Language:** Python

**Python Libraries:** Numpy, Pandas, OpenCV, Wavelet Transformation, Keras, OS, Time, Shutil, Joblib

**Machine Learning:** XgBoost, StandardScaler, Pipeline, GridSearchCV



## Deployment

- Download the whole directory
- Run:
```bash
  pip install -r requirements.txt
```
- Run "getdataset.py" to set the face-lock for yourself! 
  NOTE: Set the lighting perfect, look in camera for 30sec change orientation of your face slowly.
- Run "util.py" to use the face-lock and see the result.
- If the person is user then prints "Hello" else "Get Lost!". 



