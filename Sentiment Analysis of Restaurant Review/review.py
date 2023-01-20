import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = "\t")
df.isnull().sum()

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) # means any character that IS NOT a-z OR A-Z
    review = review.lower()     

    review = review.split()
    lm = WordNetLemmatizer()

    review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
       
    review = ' '.join(review)      
    corpus.append(review)
    
new_df = pd.concat([df,pd.DataFrame(corpus, columns=['New'])], axis = 1)

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = None)
X = cv.fit_transform(corpus).toarray() 
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn import svm
svm_clf = svm.SVC(kernel='linear')
model = svm_clf.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")

acc = accuracy_score(y_test, y_pred)

# SAMPLE ANALYSIS
def Predict_Sentiment(sample_review):
    word = []
    sample_review = re.sub('[^a-zA-Z]', ' ', sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    lm = WordNetLemmatizer()
    sample_review = [lm.lemmatize(word) for word in sample_review if not word in set(stopwords.words('english'))]
    sample_review = ' '.join(sample_review) 
    word.append(sample_review)
    temp = cv.transform(word).toarray()
    return model.predict(temp)[0]

customer_review = input("Enter the review: ")
if (Predict_Sentiment(customer_review)):
  print("Customer generated a \"Good Review\".")
else:
  print("Customer generated a \"Bad Review\".")
