from flask import  Flask,render_template,request
import joblib
import os
import numpy as n
# import pickle
# import sklearn
# import lightgbm
import xgboost
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")
@app.route("/result",methods=['POST','GET'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=n.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)
    # scaler_path=os.path.join(r'D:\DL\StrokrPrediction\ML Project Stroke\models\scalerstandard.pkl')
    # scaler=None
    # with open(scaler_path,'rb') as scaler_file:
    #     scaler = pickle.load(scaler_file)
    # x=scaler.transform(x)

    model_path=os.path.join(r'D:\DL\StrokrPrediction\ML Project Stroke\models\xgbooster-new-version-model-joblib-file.sav')
    # boost = Booster.save_model(model_path)
    dt = joblib.load(model_path,'r')

    Y_pred = dt.predict(x)
    print(Y_pred)

    if Y_pred==0:
        return  render_template("nostroke.html")
    else:
        return render_template("stroke.html")


def index():
    return render_template("home.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
#%%
