import pandas as pd
import nltk
import string
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from nltk.stem.snowball import SnowballStemmer

def preprocess(text, stem=False, tokenize=True):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.digits))
    text = nltk.word_tokenize(text)
    text = [w for w in text if w not in stopwords]
    if stem == True:
        text = [stemmer.stem(w) for w in text]
    if tokenize == False:
        text = " ".join(text)
    return text

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#df_ = df.sample(n=3000, random_state=42)
df_ = df.copy()
df_test_ = df_test.copy()
print('Sample obtained')
df_.text.fillna("", inplace=True)
df_test_.text.fillna("", inplace=True)
df_["preprocessed_text"] = [preprocess(el, tokenize=False) for el in df_.text]
df_test_["preprocessed_text"] = [preprocess(el, tokenize=False) for el in df_test_.text]
print('Text preprocessed')

tfidf = TfidfVectorizer(use_idf=True)
X_train = tfidf.fit_transform(df_.preprocessed_text)
X_test = tfidf.transform(df_test_.preprocessed_text)
print('Vectorizer run')

y_train = df_['label']

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_scores = cross_val_score(forest_reg, X_train, y_train, cv=3)

y_pred = forest_reg.predict(X_test)
label = [int(round(el)) for el in y_pred]
df_submit = pd.DataFrame({"id":list(df_test_["id"]), "label":label})
df_submit.to_csv('submit.csv', index=False)