import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv('houses_to_rent_v2.csv')
    imputer = KNNImputer(n_neighbors=3)
    df.floor = df.floor.replace('-', np.nan).astype(float)
    numeric_col = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df[numeric_col] = imputer.fit_transform(df[numeric_col])
    return df


if 'df' not in st.session_state:
    with st.spinner('Loading data...'):
        st.session_state['df'] = load_data()

st.title('Brazilian Houses')
st.header('Rent price prediction')
st.image("https://qph.cf2.quoracdn.net/main-qimg-8b7db09f0026709117d0369aeaaee360-lq")