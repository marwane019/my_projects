from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, pathlib, pandas as pd
app = FastAPI(title='Titanic Survival API', version='1.0')
_model = joblib.load(pathlib.Path('model')/'titanic_model.joblib')
_meta = json.loads((pathlib.Path('model')/'meta.json').read_text())
class Passenger(BaseModel):
    Pclass:int; Sex:str; Age:float|None=None; SibSp:int=0; Parch:int=0; Fare:float|None=None; Embarked:str='S'
@app.get('/meta') def meta(): return _meta
@app.post('/predict') def predict(p:Passenger):
    import pandas as pd
    df = pd.DataFrame([p.dict()]); proba = float(_model.predict_proba(df)[0,1]); return {'survival_probability': proba}
