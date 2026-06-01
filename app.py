import sqlite3
import contextlib
import re
import numpy as np
import pickle
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask import Flask, render_template, request, session, redirect
import os
import joblib
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.densenet import preprocess_input


# Fix: Use non-GUI backend for matplotlib to avoid RuntimeError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from create_database import setup_database
from utils import login_required, set_session

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'arivaarivaarivaarivaarivaariva'

heart = pickle.load(open('adultmodel.pkl','rb'))
childheart = pickle.load(open('childmodel.sav','rb'))
model_fetal = joblib.load('fetalmodel.pkl')
model = tf.keras.models.load_model('heart_attack_prediction_model.h5')

database = "users.db"
setup_database(name=database)

def get_className(prediction):
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "No"  # Update class names accordingly

    elif predicted_class == 1:
        return "Mild"
    elif predicted_class == 2:
        return "Moderate"
    elif predicted_class == 3:
        return "Severe"
    else:
        return "Proliferate"

INPUT_SIZE = 224  # Update the input size to match the model

def preprocess_single_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = preprocess_input(image)

    # Add an extra dimension to match the model input shape
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return prediction


@app.route('/')
def first():
    return render_template("index1.html")

@app.route('/predictupload', methods=['GET', 'POST'])
def predictupload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = preprocess_single_image(file_path)
        #result = get_className(value)
        result=''
        treatment=''
        predicted_class = np.argmax(value)
        if predicted_class == 0:
            result= "No"  # Update class names accordingly
            treatment="Treatment No Needed"

        elif predicted_class == 1:
            result= "Mild"
            treatment="prioritize attending cardiac rehabilitation, taking all prescribed medications (such as aspirin and blood thinners), and adopting a heart-healthy diet. Rest and Life Style Changes to recover."
        elif predicted_class == 2:
            result= "Moderate"
            treatment="Angioplasty and Stenting (PCI) and Clot-Busting Drugs (Thrombolytics)"
        elif predicted_class == 3:
            result= "Severe"
            treatment="Percutaneous Coronary Intervention (PCI/Angioplasty) and Coronary Artery Bypass Grafting (CABG)"
        else:
            result= "Proliferate"
            treatment="MicroRNA Therapy , Balloon angioplasty and stenting (PCI) to open blocked arteries."

       # return result
    return render_template('upload.html', prediction_text= result, treatment= treatment)

@app.route('/logout')
def logout():
    session.clear()
    session.permanent = False
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username')
    password = request.form.get('password')

    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account:
        return render_template('login.html', error='Username does not exist')

    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    if ph.check_needs_rehash(account[1]):
        query = 'update users set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    set_session(username=account[0], email=account[2], remember_me='remember-me' in request.form)
    return redirect('/adult')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()

    if result:
        return render_template('register.html', error='Username already exists')

    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {'username': username, 'password': hashed_password, 'email': email}

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            conn.execute(query, params)

    set_session(username=username, email=email)
    return redirect('/')

@app.route('/adult', methods=['GET'])
def adult():
    return render_template("adult.html")

@app.route('/child', methods=['GET'])
def child():
    return render_template("child.html")

@app.route('/upload', methods=['GET'])
def upload():
    return render_template("upload.html")

@app.route('/fetal', methods=['GET'])
def fetal():
    return render_template('fetal.html')

@app.route('/perf', methods=['GET'])
def perf():
    return render_template('perf.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        int_feature = [x for x in request.form.values()]
        final_features = [np.array(int_feature)]
        result=heart.predict(final_features)
        treatment=''
        if result == 1:
            result = "Positive"
            treatment='Consult a Professional and lifestyle Modification. Potential initiation of statins or blood pressure medication. '
        else:
            result = 'Negative'
            treatment="No treatment Needed"



        return render_template('adult.html', prediction_text= result, treatment= treatment)

    return render_template('adult.html')

@app.route("/childpredict", methods=['POST'])
def childpredict():
    if request.method == 'POST':
        int_feature = [x for x in request.form.values()]
	 
        final_features = [np.array(int_feature)]
        
        y_pred=childheart.predict(final_features)
        
        print("predicted")
        print(y_pred)
        result=""
        treat=""
        if y_pred[0]==0:
            result="Stage Normal"
            treat="dexrazoxane is no longer contraindicated"
        elif y_pred[0]==1:
            result="Stage Mild"
            treat="Adeno-associated virus gene therapy"
            
        elif y_pred[0]==2:
            result="Stage Moderate"
            treat="anti–interleukin-6 receptor antagonist such as tocilizumab "
        elif y_pred[0]==3:
            result="Stage Severe"
            treat="Immediate surgey need to given"
        else:
            result="No Disease"
            treat="No treatment Needed"

        return render_template('child.html', prediction_text= result, treatment= treat)

    return render_template('child.html')

@app.route("/fetpredict", methods=['POST'])
def fetpredict():
    if request.method == 'POST':
        baseline_value = request.form['baseline_value']
        accelerations = request.form['accelerations']
        fetal_movement = request.form['fetal_movement']
        uterine_contractions = request.form['uterine_contractions']
        light_decelerations = request.form['light_decelerations']
        severe_decelerations = request.form['severe_decelerations']
        prolonged_decelerations = request.form['prolonged_decelerations']
        abnormal_variability = request.form['abnormal_variability']
        short_variability = request.form['short_variability']
        percentage_of_variability = request.form['percentage_of_variability']
        long_variability = request.form['long_variability']
        histogram_width = request.form['histogram_width']
        histogram_min = request.form['histogram_min']
        histogram_max = request.form['histogram_max']
        histogram_of_peaks = request.form['histogram_of_peaks']
        histogram_of_zeroes = request.form['histogram_of_zeroes']
        histogram_mode = request.form['histogram_mode']
        histogram_mean = request.form['histogram_mean']
        histogram_median = request.form['histogram_median']
        histogram_variance = request.form['histogram_variance']
        histogram_tendency = request.form['histogram_tendency']

        data = {
            'baseline value': baseline_value,
            'accelerations': accelerations,
            'fetal_movement': fetal_movement,
            'uterine_contractions': uterine_contractions,
            'light_decelerations': light_decelerations,
            'severe_decelerations': severe_decelerations,
            'prolongued_decelerations': prolonged_decelerations,
            'abnormal_short_term_variability': abnormal_variability,
            'mean_value_of_short_term_variability': short_variability,
            'percentage_of_time_with_abnormal_long_term_variability': percentage_of_variability,
            'mean_value_of_long_term_variability': long_variability,
            'histogram_width': histogram_width,
            'histogram_min': histogram_min,
            'histogram_max': histogram_max,
            'histogram_number_of_peaks': histogram_of_peaks,
            'histogram_number_of_zeroes': histogram_of_zeroes,
            'histogram_mode': histogram_mode,
            'histogram_mean': histogram_mean,
            'histogram_median': histogram_median,
            'histogram_variance': histogram_variance,
            'histogram_tendency': histogram_tendency
        }
        input_data = pd.DataFrame([data])
        result=""
        prediction = model_fetal.predict(input_data)[0]
        if prediction==1:
            result="Normal"
            treat="No treatment Needed"
        elif prediction==2:
            result="Suspect"
            treat="Keep the person calm and seated, give them an aspirin to chew (if not allergic), and, if they are unconscious and not breathing, begin CPR. "
            
        elif prediction==3:
            result="Pathological"
            treat="Restoring normal rhythm or rate using medications (beta-blockers, blood thinners), lifestyle changes, or procedures. "
       
        else:
            result="No Disease"
            treat=""


        return render_template('fetal.html', prediction_text= result, treatment= treat)

    return render_template('fetal.html')



if __name__ == "__main__":
    app.run(debug=True)

