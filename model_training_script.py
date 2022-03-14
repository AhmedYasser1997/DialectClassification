import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
import keras.preprocessing.text
import keras.backend as K
import pickle
from keras.metrics import Precision , Recall , Accuracy , TruePositives , TrueNegatives , FalsePositives , FalseNegatives

nltk.download('stopwords')
df = pd.read_csv('final_dialect_dataset.csv')
df, df_extra = train_test_split(df, test_size=0.5, stratify=df['dialect'])
# The text will be our training independent x variable and the dialect is our dependent y variable
X = df.iloc[:, 1].apply(lambda x: np.str_(x)).values
y = df.iloc[:, 2].values

def tokenize(text):
    """
    Tokenize the text function and remove stop words (Will be passed as a parameter to the count vectorizer)
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        no_stop_words -> List of tokens from the provided text
    """
    tokens = word_tokenize(text)
    no_stop_words = [word for word in tokens if word not in stopwords.words('arabic')]
    return no_stop_words

# Perform a train-test split (80% training & 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Now we will build a pipeline that take the features from our text data using count tokens (CountVectorizer) and tf-idf scores (TfidfTransformer) and then it will be passed to a multinomial naive bayes classifier which will give us the prediction.
pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize, max_df = 0.8)),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
pickle.dump(pipeline, open('classifier.pkl', "wb"))