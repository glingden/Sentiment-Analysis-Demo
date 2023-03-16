"""
This app predicts the given text into one of 3 classes: 1.postive, 2.neutral and , 3. Negative
Author: Ganga Lingden

"""

import pickle
import random

from flask import Flask,render_template,url_for,request
from  text_processing import  text_cleaning


# Load model
tfidf_object =  pickle.load(open('Models/tfidf_vect', 'rb')) # TFIDF vectorizer
model =  pickle.load(open('Models/final_logistic', 'rb')) # Predicted model

# Randomly select between two options
try_again_options = ["Let's try again.", "Do you want to try again?"]
try_again = random.choice(try_again_options)


#  Creates an instance of the Flask App
app = Flask(__name__)


@app.route('/')
def home():
    """Renders the home page of the web app."""

    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Predicts the sentiment of the given input text."""

    if request.method == 'POST':
        text = request.form['comment']
    
        # Check if input text is provided
        if text:
            # Clean text
            clean_text = text_cleaning(text) 

            # Convert cleaned text into a matrix using tf-idf vectorizer
            matrix = tfidf_object.transform([clean_text])

            # Predict the sentiment using trained model
            predict_value = model.predict(matrix) 

            return render_template('result.html', 
                                   prediction=predict_value,
                                    try_again = try_again)
        
        else:
            return render_template('result.html', 
                                    prediction='Please, Enter some text to analyse !!!',
                                    try_again = try_again)
        




if __name__ == '__main__':
    app.run(debug=True)


