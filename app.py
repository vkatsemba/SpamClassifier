from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv",encoding='latin-1')
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

    df['label'] = df['v1'].map({'ham': 0, 'spam': 1}).astype(int)
    X = df['v2']
    y = df['label']

    cv = CountVectorizer()
    X = cv.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    from sklearn.linear_model import SGDClassifier

    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    sgd.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = sgd.predict(vect)
    return render_template('Result.html',prediction = my_prediction)
    
if __name__ == '__main__':
    app.run(debug=True)        