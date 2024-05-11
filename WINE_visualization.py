import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
import pandas as pd
from sklearn.datasets import load_wine


st.title('WINE Data visualization')
st.subheader('PCA')
#df

tmp =  load_wine(as_frame=True)
df = tmp['data']
df['target'] = tmp['target']




#PCA
d = st.select_slider("Select PCA dimension",options=[1,2,3])

if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :13])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='target')

    st.plotly_chart(fig)

if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

#MDS
st.subheader('MDS')
d = st.select_slider("Select MDS dimension",options=[1,2,3])
if d == 3:
    X = MDS(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='target')

    st.plotly_chart(fig)

if d == 2:
    X = MDS(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

if d == 1:
    X = MDS(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)


#Isomap
st.subheader('Isomap')

d = st.select_slider("Select Isomap dimension",options=[1,2,3])
k = st.select_slider("Select Isomap neighbors",options=range(3,20))
if d == 3:
    X = Isomap(n_components=3 , n_neighbors= k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='target')

    st.plotly_chart(fig)

if d == 2:
    X = Isomap(n_components=2 , n_neighbors= k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

if d == 1:
    X = Isomap(n_components=1 , n_neighbors= k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

#TSNE
st.subheader('T-SNE')
d = st.select_slider("Select TSNE dimension",options=[1,2,3])
if d == 3:
    X = TSNE(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y','z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='target')

    st.plotly_chart(fig)

if d == 2:
    X = TSNE(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x','y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

if d == 1:
    X = TSNE(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                        x='x',
                        y='y',
                        color='target')

    st.plotly_chart(fig)

