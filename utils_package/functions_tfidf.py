#for preprocessing
import re
import nltk

from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
#! python -m spacy download en_core_web_lg

def clean_data(text,
               unicode_char=True,
               specific_rules=True,
               number=True,
               case=True,
               strip=True,
               html=True,
               punctuation=True
               ):

    if type(text) != 'str':
        text = str(text)

    if unicode_char:
        text = text.encode("ascii", "ignore").decode()

    if specific_rules:
        text = text.replace("h&gt",'').replace("&lt",'').replace("&gt",'')

    if number:
      text = re.sub('\d+', '', text)

    if case:
        text = text.lower()

    if strip:
        text = text.strip()

    if html:
        text = re.sub('<[^<]+?>', ' ', text)

    if punctuation:
        text = re.sub('[^\\w\\s#]', ' ', text)

    return text

def remove_stop_word(text):
    stopwords = list(set(sw.words('english')))
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def tokenize(text):
  tokenizer = nltk.RegexpTokenizer(r'\w+')
  text = tokenizer.tokenize(text)
  return text

def final_cleaning(question, token=True):
    final_question = clean_data(question)
    final_question = remove_stop_word(final_question)
    final_question = lemmatizing(final_question)
    if token == True:
        final_question = tokenize(final_question)
    return final_question

