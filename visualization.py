import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

st.title('Data visualization')
df = px.data.iris()
#df
#PCA
d = st.select_slider("Select PCA dimension",options=[1,2,3])

if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['species'] = df['species']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='species')

    st.plotly_chart(fig)

if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['species'] = df['species']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='species')

    st.plotly_chart(fig)

if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['species'] = df['species']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='species')

    st.plotly_chart(fig)