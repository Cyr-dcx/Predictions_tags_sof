#pour l'api
# import os
# from pathlib import Path
# from typing import List
from fastapi import FastAPI #HTTPException
from pydantic import BaseModel
from model.model import predict_pipeline

# pour la modelisation
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# import lightgbm as lgb

app = FastAPI()

class Phrase(BaseModel):
    phrase: str

# class Tags(BaseModel):
#     tags: List[str]

sentence_test="I've been making Python scripts for simple tasks at work and never really bothered packaging them for others to use. Now I have been assigned to make a Python wrapper for a REST API. I have absolutely no idea on how to start and I need help.What I have:(Just want to be specific as possible) I have the virtualenv ready, it's also up in github, the .gitignore file for python is there as well, plus, the requests library for interacting with the REST API. That's it.Here's the current directory tree"

@app.get("/")
def index():
    return {"tags": "Faisons une prédiction"}

@app.post("/predict", status_code=200)
def read_item(one_phrase: Phrase):
    question = one_phrase.phrase
    tags = predict_pipeline(question)

    return {"tags": tags}
