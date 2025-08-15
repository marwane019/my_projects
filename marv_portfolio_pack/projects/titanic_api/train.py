# Train a simple pipeline for Titanic with robust preprocessing
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib, json, pathlib

def main():
    df = pd.read_csv('train.csv')
    y = df['Survived']
    X = df.drop(columns=['Survived','Name','Ticket','Cabin'])
    num = ['Age','SibSp','Parch','Fare']; cat = ['Pclass','Sex','Embarked']
    pipe = Pipeline([('pre', ColumnTransformer([
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), num),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('oh', OneHotEncoder(handle_unknown='ignore'))]), cat)
            ])),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1))
        ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    pipe.fit(X, y)
    pathlib.Path('model').mkdir(exist_ok=True)
    joblib.dump(pipe, 'model/titanic_model.joblib')
    (pathlib.Path('model')/'meta.json').write_text(json.dumps({'cv_mean_acc':float(scores.mean()),'cv_std':float(scores.std()),'features':num+cat}, indent=2))
    print('Saved model. CV acc:', float(scores.mean()))
if __name__=='__main__': main()
