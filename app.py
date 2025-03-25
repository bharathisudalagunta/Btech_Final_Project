from flask import Flask, render_template,url_for,request,jsonify
import joblib
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def result():
    d1=int(request.form['a'])
    d2=int(request.form['b'])
    d3=int(request.form['c'])
    d4=int(request.form['d'])
    d5=int(request.form['e'])
    d6=int(request.form['f'])
    d7=int(request.form['g'])
    values=[[d1,d2,d3,d4,d5,d6,d7]]
    sc=StandardScaler()
    values=sc.fit_transform(values)
    model=pickle.load(open('model_pic','rb'))
    #model=load_model(r"C:\Users\bhara\OneDrive\Desktop\flask_demo\model_pic")
    prediction=model.predict(values)
    prediction=float(prediction)
    print(prediction)
    
    json_dict={
        "prediction":prediction
    }   
    
    return jsonify(json_dict)

if __name__ == "__main__":
    app.run(debug=True)