"""
This app predicts the given text into 3 class 1.postive, 2.neutral and , 3. Negative
Author: Ganga Lingden

"""


from flask import Flask,render_template,url_for,request
from  text_processing import  text_cleaning
from langdetect import detect
import pickle

#load models
tfidf_object =  pickle.load(open('Models/tfidf_vect', 'rb')) #tfidf vectorizer
model =  pickle.load(open('Models/final_logistic', 'rb')) # predicted model


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():


    if request.method == 'POST':

        text = request.form['comment']

        #check if get some value
        if text:

            #check language
            if len(text.split()) >3:
                if detect(text) == 'en':
                    clean_text = text_cleaning(text) #clean text
                    print(clean_text)
                    matrix = tfidf_object.transform([clean_text]) #convet into matrix
                    predict_value = model.predict(matrix) # predict sentiment
                    return render_template('result.html', prediction=predict_value)

                else:
                    return render_template('result.html', prediction='Only work for English Lanuage !!')
            else:
                clean_text = text_cleaning(text)  # clean text
                print(clean_text)
                matrix = tfidf_object.transform([clean_text])  # convet into matrix
                predict_value = model.predict(matrix)  # predict sentiment
                return render_template('result.html', prediction=predict_value)


        else:
            return render_template('result.html', prediction='Please, Enter some text(reviews) !!')


    #if want to display result in home.html
    #return render_template('home.html', prediction='Predicted Sentiment: '+ clean_text)

if __name__ == '__main__':
    app.run(debug=True)


