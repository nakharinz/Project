import streamlit as st
import pandas as pd

#read df from csv iris dataset from UCI url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']

#Hello world
st.title('Hello World')
st.write('This is Iris dataset from UCI')
st.dataframe(df)
