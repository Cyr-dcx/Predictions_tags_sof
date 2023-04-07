import pandas as pd
import numpy as np


def get_embed():
    import tensorflow_hub as hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return embed

def feature_USE_fct(sentences, b_size=1) :
    if type(sentences) != list:
        sentences = [''.join(list(sentences))]

    batch_size = b_size
    # time1 = time.time()
    embed = get_embed()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    # time2 = np.round(time.time() - time1,0)
    return features

# sentence= "I've been making Python scripts for simple tasks at work and never really bothered packaging them for others to use. Now I have been assigned to make a Python wrapper for a REST API. I have absolutely no idea on how to start and I need help.What I have:(Just want to be specific as possible) I have the virtualenv ready, it's also up in github, the .gitignore file for python is there as well, plus, the requests library for interacting with the REST API. That's it.Here's the current directory tree"
# test = feature_USE_fct(list(sentence), 1)
# print(test)
