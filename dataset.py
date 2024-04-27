import seaborn as sns
import streamlit as st

st.title('My Dataset')

ds = st.selectbox('Select Dataset',sns.get_dataset_names())

df = sns.load_dataset(ds)

df


