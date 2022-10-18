import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

msg_data = pd.read_csv('Z:/smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
lemm = WordNetLemmatizer()
review = []
for i in range(0, len(msg_data)):
    words = re.sub('[^a-zA-Z]',' ',msg_data['message'][i])
    words = words.lower()
    words = words.split()

    words = [lemm.lemmatize(word) for word in words if not word in stopwords.words('english')]   
    words = ' '.join(words)
    review.append(words)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(review).toarray()
y = pd.get_dummies(msg_data['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
