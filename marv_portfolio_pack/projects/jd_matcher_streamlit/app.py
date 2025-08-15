import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
st.set_page_config(page_title='JD ↔ CV Matcher', layout='wide')
st.title('JD ↔ CV Matcher'); st.caption('Compute similarity & missing keywords.')
jd = st.text_area('Job Description', height=220); cv = st.text_area('Your CV (paste text)', height=220)
def normalize(t):
    t=t.lower(); t=re.sub(r'[^a-z0-9\s\-\+\.#]', ' ', t); return re.sub(r'\s+',' ',t).strip()
if st.button('Analyze') and jd and cv:
    docs = [normalize(jd), normalize(cv)]
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english')
    try:
        X = v.fit_transform(docs); sim = cosine_similarity(X[0], X[1])[0,0]
    except ValueError:
        v = TfidfVectorizer(ngram_range=(1,1), min_df=1, stop_words='english'); X = v.fit_transform(docs); sim = cosine_similarity(X[0], X[1])[0,0]
    st.metric('Similarity', f'{sim*100:.1f}%')
    jd_terms = set([t for t in v.get_feature_names_out() if t in docs[0]])
    cv_terms = set([t for t in v.get_feature_names_out() if t in docs[1]])
    missing = sorted(list(jd_terms - cv_terms))[:60]
    st.subheader('Top Missing Keywords'); st.write(', '.join(missing) if missing else 'Great — you cover most key terms!')
