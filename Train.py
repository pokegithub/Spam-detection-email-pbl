import pandas as pd
import joblib
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import VotingClassifier
from nltk.stem import SnowballStemmer
# 1. Load Data, currently since I was using kaggle cloud service so the csv location is a bit off and I will change it if needed.
print("Loading dataset...")
df = pd.read_csv('/kaggle/input/datasets/bhaskarbahukhandi/sms-spam-collection-dataset/data.csv', encoding='latin-1')
# 2. Preprocessing
stemmer = SnowballStemmer("english")
def clean(text):
    # Lowercase, remove punctuation, and stem in one optimized pass, this is the method you told me to do
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return " ".join([stemmer.stem(w) for w in text.split()])
df['text'] = df['text'].apply(clean)
# 3. This is the train_test_split that I talked about in college. x_tr is training text, y_tr are training answers, x_te is the test text (the questions) and y_te are the test answers
x_tr, x_te, y_tr, y_te = train_test_split(df['text'], df['label'], test_size=0.3, random_state=0)
#test_size=0.3 means 30% is taken for test just like you asked. random_state is just there so that same the data points will be assigned to the training and testing sets every time someone executes the code.
vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) #This is vectorizer and stop_words='english' means we basically ignore stop words.
x_tr_v = vec.fit_transform(x_tr)
x_te_v = vec.transform(x_te)
# Keep top 1000 features, btw this is a filter method of feature selection so you can look it up to understand more about it.
sel = SelectKBest(chi2, k=1000) 
x_tr_s = sel.fit_transform(x_tr_v, y_tr)
x_te_s = sel.transform(x_te_v)
# 4. Model Tuning, we will use this to get the best model to get the best accuracy
models = [{'name': 'LR', 'est': LogisticRegression(), 'grid': {'C': [0.1, 1, 10, 100]}},
          {'name': 'SVM', 'est': SVC(probability=True), 'grid': {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}}]
best_mdls = {}
x_samp, y_samp = x_tr_s[:2000], y_tr[:2000]
for m in models:
    grid = GridSearchCV(m['est'], m['grid'], cv=2, n_jobs=-1)
    grid.fit(x_samp, y_samp)
    best_mdls[m['name']] = grid.best_estimator_
ensemble = VotingClassifier(
    estimators=[('lr', best_mdls['LR']), ('svm', best_mdls['SVM'])],
    voting='soft')
# Training in 70K like you asked
ensemble.fit(x_tr_s[:70000], y_tr[:70000])
y_pred = ensemble.predict(x_te_s)
y_prob = ensemble.predict_proba(x_te_s) # Probabilities needed for Log Loss
acc = accuracy_score(y_te, y_pred)
loss = log_loss(y_te, y_prob)
print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"Log Loss: {loss:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
# 7. Saving my progress :)
joblib.dump(ensemble, '/kaggle/working/model.pkl')
joblib.dump(vec, '/kaggle/working/vec.pkl')
joblib.dump(sel, '/kaggle/working/sel.pkl')
#Obviously as I told you before that since I was training the model in kaggle so the files saving was also set for kaggle.
#I will change this later
