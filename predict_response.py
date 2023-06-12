#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# words to be igonred/omitted while framing the dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

       # Tokenize the user_input
    tokens = nltk.word_tokenize(user_input)

    # Convert the user input into its root words: stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Remove duplicacy and sort the user_input
    unique_tokens = list(set(stemmed_tokens))
    sorted_tokens = sorted(unique_tokens)

    # Input data encoding: Create BOW for user_input
    for token in sorted_tokens:
        if token in stemmed_tokens:
            bag.append(1)
        else:
            bag.append(0)

    return np.array(bag)
    
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    
    # extract the class from the predicted_class_label
    predicted_class = predicted_class_label['class']
    
    # now we have the predicted tag, select a random response
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            # choose a random bot response
            bot_responses = intent['responses']
            bot_response = random.choice(bot_responses)
            
            return bot_response
    

print("Hi I am Stella, How Can I help you?")

while True:

    # take input from the user
    user_input = input('Type you message here : ')

    response = bot_response(user_input)
    print("Bot Response: ", response)