import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

from sklearn.datasets import load_diabetes
tmp = diabetes = load_diabetes(as_frame=True)
df = tmp['data']
df['color'] = tmp['target']
st.title('Data visualization')
#df
#PCA
d = st.select_slider("Select PCA dimension",options=[1,2,3])

if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['color'] = df['color']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='color'
                        )

    st.plotly_chart(fig)

if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['color'] = df['color']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='color'
                       )

    st.plotly_chart(fig)

if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['color'] = df['color']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='color'
                     )

    st.plotly_chart(fig)