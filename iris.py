import seaborn as sns
import streamlit as st

st.title('Iris dataset')
df = sns.load_dataset('iris')

x = st.selectbox('Select X-axis', df.columns[:-1])
y = st.selectbox('Select Y-axis', df.columns[:-1])

st.write('You seleceted:',x,y)

st.scatter_chart(df, x=x, y=y, color=df.columns[-1])