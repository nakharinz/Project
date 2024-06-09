import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv('_insurance_data.csv')

    #One-hot encoding on 'sex' col
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    df['sex'] = encoder.fit_transform(df[['sex']])
    df.rename(columns={'sex': 'sex_male'}, inplace=True)

    #Fix 0 blood pressure value
    df.bloodpressure = df.bloodpressure.replace(0, np.nan).astype(float)

    imputer = KNNImputer(n_neighbors=3)
    numeric_col = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df[numeric_col] = imputer.fit_transform(df[numeric_col])
    return df


if 'df' not in st.session_state:
    with st.spinner('Loading data...'):
        st.session_state['df'] = load_data()

st.title('')
st.header('Insurance Claim Prediction')
st.image("Insurance.jfif")