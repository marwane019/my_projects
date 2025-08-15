# Titanic API

- Train: `python train.py` (needs Kaggle `train.csv`)
- Serve: `uvicorn app:app --reload`
- Docker: `docker build -t titanic-api . && docker run -p 8000:8000 titanic-api`
