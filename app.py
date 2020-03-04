import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('Log.pkl', 'rb'))

df = pd.read_csv("Modified_Errors.csv",engine='python')
'''
print(df)
'''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Jira_Summary).toarray()
    #labels = df.category_id
    features.shape

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    X_train, X_test, y_train, y_test = train_test_split(df['Jira_Summary'], df['Bug _Type'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    int_features = [str(x) for x in request.form.values()]
    #print('Bug_Type',clf.predict(count_vect.transform(["AWS-Prod(Iedp_cmps)-mpc-rds.cd40qupvbq4x.us-west-2.rds.amazonaws.com,1433 User ID=ITGReadWriteUser; Password=Write.Black.*8283.User@123; "])))
    #Fitting model with trainig data
    prediction = model.predict(count_vect.transform(int_features))
    #prediction = model.predict(int_features)
    #print(prediction)

    ##output = prediction[0]

    return render_template('index.html', prediction_text='Here is your Bug Type:: {}'.format(prediction))



# Loading model to compare the results
    #model = pickle.load(open('Log.pkl','rb'))



if __name__ == "__main__":
    app.run(debug=True)