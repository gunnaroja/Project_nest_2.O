from flask import Flask, request, render_template
import pickle
import nltk
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TfidfVectorizer
with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Load emoji pattern and stopwords
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F700-\U0001F77F"  
                           u"\U0001F780-\U0001F7FF" 
                           u"\U0001F800-\U0001F8FF"  
                           u"\U0001F900-\U0001F9FF"  
                           u"\U0001FA00-\U0001FA6F"  
                           u"\U0001FA70-\U0001FAFF"  
                           "]+", flags=re.UNICODE)

# Load and preprocess the data
data = pd.read_csv("Train.csv", encoding='ISO.8859-1')
column_mapping = {'ï»¿text': 'text'}
data = data.rename(columns=column_mapping)

# Fit the TfidfVectorizer on the data
x = tfidf.fit_transform(data['text'])
y = data.label.values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

# Create and fit the LogisticRegressionCV model
clf = LogisticRegressionCV(cv=5)
clf.fit(x_train, y_train)

app = Flask(__name__)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text = ' '.join(emojis).replace('-', '') + re.sub('[\W+]', ' ', text).lower()
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = preprocessing(comment)
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = clf.predict(comment_vector)[0]

        
        return render_template('index.html', prediction=prediction)

if __name__== "__main__":
    app.run(debug=True)