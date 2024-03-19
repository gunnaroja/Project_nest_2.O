import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

data = pd.read_csv("Train.csv",encoding = 'ISO.8859-1')
data['label'].value_counts()
column_mapping = {'ï»¿text': 'text'}
data = data.rename(columns=column_mapping)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwords_set=set(stopwords.words('english'))
emoji_pattren=re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

def preprocessing(text):
  text=re.sub('<[^>]*>', '', text)
  emojis=emoji_pattren.findall(text)
  text=re.sub('[\W+]',' ',text.lower()) + ' '.join(emojis).replace('-','')
  porter=PorterStemmer()
  text=[porter.stem(word) for word in text.split() if word not in stopwords_set]
  return " ".join(text)
preprocessing('this is my tags <h1> :) <p>helo world<p> <div> <div> </h2')
data['text']=data['text'].apply(lambda x: preprocessing(x))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,use_idf=True,norm=None,smooth_idf=True)
y=data.label.values
x=tfidf.fit_transform(data.text)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5)
from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(x_train,y_train)
y_pred= clf.predict(x_test)
pickle.dump 
import pickle

# Assuming you have some data to pickle
data_to_pickle = {'key': 'value'}

# Specify the file path where you want to save the pickled data
file_path = 'data.pkl'

# Open the file in binary mode and use pickle.dump to save the data
with open(file_path, 'wb') as file:
    pickle.dump(data_to_pickle, file)

(clf,open('clf.pkl','wb'))
pickle.dump(tfidf,open('tfidf.pkl','wb'))
def prediction(comment):
  preprocessed_comment=preprocessing(comment)
  comment_list=[preprocessed_comment]
  comment_vector=tfidf.transform(comment_list)
  prediction=clf.predict(comment_vector)[0]
  return prediction
if prediction==1:
  print("positive comment")
else:
  print("negative comment")