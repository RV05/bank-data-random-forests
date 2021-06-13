from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)


pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return(str(prediction))

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return(str(list(prediction)))
# http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=2


#sudo docker run -p 5000:5000 money_api1



if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')