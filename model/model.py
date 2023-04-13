import joblib

# pour le preprocessing de la question
from utils_package.functions_tfidf import final_cleaning
import sys

encoder_file = "./target_encoder.sav"
tfidf_file = "./tfidf_encoder.sav"
model_file = "./tfidf_lightGBM.sav"

target_encoder = joblib.load(encoder_file)
tfidf = joblib.load(tfidf_file)
model = joblib.load(model_file)

print(sys.getsizeof(target_encodergit))
print(sys.getsizeof(tfidf))
print(sys.getsizeof(model))

def preprocess_pipeline(question):
    question_list = []
    preprocessed_question = final_cleaning(question, token=False)
    question_list.append(str(preprocessed_question))
    X_tfidf = tfidf.transform(question_list)
    return X_tfidf

def generate_prediction(preprocessed_question, my_model=model):
    tags = my_model.predict(preprocessed_question)
    return tags

def predict_pipeline(question,target_encoder=target_encoder):
    preprocessed_question = preprocess_pipeline(question)
    predictions = generate_prediction(preprocessed_question)
    tags = target_encoder.inverse_transform(predictions)
    return tags
    