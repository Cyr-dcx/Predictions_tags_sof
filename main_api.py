import joblib
# import os
# from pathlib import Path
from typing import List, Union
from fastapi import FastAPI 
from pydantic import BaseModel
from utils_package.functions import *
# from sklearn.preprocessing import StandardScaler

app = FastAPI()

class Phrase(BaseModel):
    phrase: Union[str,int]

encoder_file = "./target_encoder.sav"
scaler_file = "./scaler_USE.sav"
model_file = "./xgboost_USE.sav"

target_encoder = joblib.load(encoder_file)
scaler = joblib.load(scaler_file)
model = joblib.load(model_file)

# %%
sentence_test="I've been making Python scripts for simple tasks at work and never really bothered packaging them for others to use. Now I have been assigned to make a Python wrapper for a REST API. I have absolutely no idea on how to start and I need help.What I have:(Just want to be specific as possible) I have the virtualenv ready, it's also up in github, the .gitignore file for python is there as well, plus, the requests library for interacting with the REST API. That's it.Here's the current directory tree"

def preprocess_pipeline(question, scaler=scaler):
    # preprocessed_question = final_cleaning(question, token=False)
    features_USE = feature_USE_fct(question, b_size=1)
    X_processed = scaler.transform(features_USE)
    return X_processed

def generate_prediction(preprocessed_question, my_model=model):
    tags = my_model.predict(preprocessed_question)
    return tags

test = "ko"
if test == "ok":
    X = preprocess_pipeline(sentence_test)
    test_predict = model.predict(X)
    print(test_predict)
    print(test_predict.shape)
    print(len(sentence_test))

# @app.get("/")
# def say_hello():
#     return {"hello": "word"}

@app.post("/predict", status_code=200)
def read_item(one_phrase: Phrase):
    question = one_phrase.phrase
    preprocessed_question = preprocess_pipeline(question)
    predictions = generate_prediction(preprocessed_question, my_model=model)
    tags = target_encoder.inverse_transform(predictions)

    return {"tags": tags}
