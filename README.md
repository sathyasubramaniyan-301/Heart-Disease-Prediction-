# An Effective Health Recommendation System for Cardiovascular Disease Using Machine Learning

## Overview

This project presents an intelligent cardiovascular disease prediction and health recommendation system that combines structured clinical data and retinal image analysis to improve diagnostic accuracy. The system leverages Machine Learning and Deep Learning techniques to identify heart disease risk and provide personalized health recommendations.

## Features

* Heart disease prediction using clinical datasets
* Retina image-based cardiovascular risk analysis
* Support for Adult, Child, and Fetal datasets
* Gradient Boosting-based classification model
* CNN-based retinal image feature extraction
* Stage-wise disease prediction (Normal, Early, Moderate, Severe)
* Personalized health recommendations
* Real-time prediction and analysis

## Technologies Used

### Programming Languages

* Python
* JavaScript
* HTML
* CSS

### Machine Learning & Deep Learning

* Gradient Boosting (XGBoost)
* Convolutional Neural Network (CNN)
* Scikit-learn
* TensorFlow / Keras

### Libraries & Tools

* NumPy
* Pandas
* OpenCV
* Flask
* Joblib
* Pickle
* SQLite
* VS Code

## System Architecture

The system follows a hybrid architecture:

1. Data Collection

   * Adult Dataset (CSV)
   * Child Dataset (CSV)
   * Fetal Dataset (CSV)
   * Retina Images

2. Data Preprocessing

   * Missing value handling
   * Data normalization
   * Feature encoding
   * Image enhancement and resizing

3. Feature Extraction

   * Clinical feature selection
   * CNN-based retinal image feature extraction

4. Model Training

   * Gradient Boosting for structured data
   * CNN for image analysis

5. Prediction & Classification

   * Normal
   * Early Stage
   * Moderate Stage
   * Severe Stage

6. Recommendation Engine

   * Lifestyle modification suggestions
   * Medical consultation recommendations
   * Preventive healthcare guidance

## Project Modules

### Data Acquisition

Collects patient data and retinal images.

### Data Preprocessing

Cleans and transforms structured and image data.

### Feature Engineering

Extracts important clinical and visual features.

### Model Training

Trains Gradient Boosting and CNN models.

### Prediction Module

Predicts heart disease severity levels.

### Recommendation Module

Generates personalized health recommendations.

### Evaluation Module

Measures model performance using:

* Accuracy
* Precision
* Recall
* F1-Score

## Results

The proposed system achieved high prediction performance with an accuracy ranging between 92% and 96%. The integration of retinal image analysis with structured clinical data significantly improved prediction reliability compared to traditional approaches.

## Future Enhancements

* Cloud deployment using AWS
* Mobile application integration
* Explainable AI (XAI)
* Electronic Health Record (EHR) integration
* Real-time patient monitoring

## Project Structure

```bash
Heart-Disease-Recommendation-System/
│
├── Dataset/
│   ├── Adult.csv
│   ├── Child.csv
│   ├── Fetal.csv
│   └── Retina_Images/
│
├── Models/
│   ├── adultmodel.pkl
│   ├── childmodel.sav
│   └── fetalmodel.pkl
│
├── Templates/
│   ├── login.html
│   ├── register.html
│   ├── adult.html
│   ├── child.html
│   └── fetal.html
│
├── Static/
│
├── app.py
├── requirements.txt
└── README.md
```

## Authors

* Sathya S
* Suryapriya P
* Bavithra V

## Institution

Department of Computer Science and Engineering

University College of Engineering, Thirukkuvalai

Anna University, Chennai

## License

This project is developed for academic and research purposes.
## Research Paper
[View Research Paper](https://ijpub.org/ijvra/papers/IJVRA2605254.pdf)
